"""
Helper function to get content passed with http request.
"""
function GetContent(request::HTTP.Request)
    content = JSON.parse(String(request.body))
    return content
end

"""
Sends a response containing a json object (usually the action or fitness value).
"""
function PostResponse(response::Dict{String}{Int64})
    return HTTP.Response(200,JSON.json(response))
end

"""
ServerHandler used for handling the different service:
- getting features and returning action.
- getting the nbKill, nbDeath and earlyPenalty values
- getting the ratioDamageTowerOpp, totalDamageToOpp values
- handle early stopping
- close the server when a game is over
"""
function ServerHandler(request::HTTP.Request)
    global lastFeatures
    global oldLastFeatures
    global individual
    global breezyIp
    global breezyPort
    global server
    global nbKill
    global nbDeath
    global earlyPenalty
    global totalDamageToOpp
    global ratioDamageTowerOpp

    path = HTTP.URIs.splitpath(request.target)
    # path is either an array containing "update" or nothing so the following line means "if there is an update"
    if (size(path)[1] != 0)
        # Update route is called, game finished.

        cfg["n_game"] += 1
        content = GetContent(request)
        # println(content)
        if (content["status"] == "DONE")
            nbDeath = content["deaths"]
            nbKill = content["radiantKills"]
        elseif (content["status"] == "CANCELED") # i.e EarlyStop
            earlyPenalty = 1
        end

        # now the game is over we can get the ratioDamageTowerOpp
        ratioDamageTowerOpp = GetTowerRatio(lastFeatures)

        # Since the Game is over we want to close the server
        # closing the server generate an error, in order to keep the code running we use a try & catch
        try
            close(server)
        catch e
            return HTTP.post(404,JSON.json(Dict("socket closed"=>"0")))
        end

    else
        # Relay route is called, gives features from the game for the agent.
        # get data as json, then save to list
        content = GetContent(request)
        # features = JSON.json(content)
        # println(features)
        oldLastFeatures = lastFeatures
        lastFeatures = content
        # you need this conversion to call process
        lastFeatures = convert(Array{Float64},lastFeatures)
        # from here we can get the damage made between two state
        totalDamageToOpp += EstimateDamage(oldLastFeatures,lastFeatures)

        if EarlyStop(lastFeatures)
            # EarlyStop will stopped the current game by calling the upgrade route
            # we send to the Breezy server to call the update route
            stopUrl = "http://$breezyIp:$breezyPort/run/active"
            response = HTTP.delete(stopUrl)
        else
            # Agent code to determine action from features.
            # julia array start at 1 but breezy server is python so you need the "-1"
            inputs = get_inputs(lastFeatures)
            action = argmax(process(individual, inputs)) - 1
            PostResponse(Dict("actionCode"=>action))
        end
    end
end

"""
This function allow us to:
- play one game and to get the fitness score(as an Array for Cambrian requirements) of one individual
- update lastFeatures, oldLastFeatures, totalDamageToOpp, ratioDamageTowerOpp corresponding to the individual
- update nbKill, nbDeath, earlyPenalty corresponding to the game just played
"""
function PlayDota(ind::CGPInd)
    global server
    global breezyIp
    global breezyPort
    global agentIp
    global agentPort
    global individual
    global nbKill
    global nbDeath
    global earlyPenalty
    global lastFeatures
    global oldLastFeatures
    global totalDamageToOpp
    global ratioDamageTowerOpp

    # set game variables
    nbDeath = 0
    nbKill = 0
    earlyPenalty = 0
    # set the global variable (the one Handler can manage) to the individual you want to evaluate
    individual = ind
    # initialize the server
    server = Sockets.listen(Sockets.InetAddr(parse(IPAddr,agentIp),parse(Int64,agentPort)))
    # the url we need to trigger to start a game
    startUrl = "http://$breezyIp:$breezyPort/run/"
    # initialize game
    response = HTTP.post(startUrl, ["Content-Type" => "application/json"], JSON.json(startData))
    # initialize lastFeatures and oldLastFeatures
    # lastFeatures = [0.0]
    oldLastFeatures = [0.0]
    # initialize the damage variables
    totalDamageToOpp = 0
    ratioDamageTowerOpp = 0
    # will run the game until it is over, when it is over there is error because of the server closure
    try
        HTTP.serve(ServerHandler,args["agentIp"],parse(Int64,args["agentPort"]);server=server)
    # when there is the error we know the game is over and we can return the fitness
    catch e
        [Fitness1(lastFeatures,nbKill,nbDeath,earlyPenalty)]
    end
end

"""
This function return the coordinate in the behavior space of individual.
This is required to use MapElites. Feel free to try another characterization.
Needs to return an Array{Int64}
"""
function MapIndToB(mapArray)

    x = deepcopy(mapArray[1])
    y = deepcopy(mapArray[2])
    y_ind = 1 + Int(round(100*y/2))

    if y_ind > 50
        y_ind = 50
    end

    if x <= 800
        x_ind = 1 + Int(round(x/40))
    elseif x <= 4000
        x_ind = 20 + Int(round((x-800)/110))
    else
        x_ind = 50
    end

    [x_ind,y_ind]
end

"""
Modification of MapElites to adapt it to DOTA2 problem
Notes: original function in "MapElites/src/poulate_function.jl"
"""
function MapelitesDotaStep!(e::Evolution,
                             map_el::MAPElites.MapElites,
                             map_ind_to_b::Function;
                             mutation::Function=Cambrian.uniform_mutation,
                             crossover::Function=Cambrian.uniform_crossover,
                             evaluate::Function=Cambrian.random_evaluate)

    # the MappingArray is at first empty and is filled when evaluate is called,
    # it will be used to get MapElites coordinates
    global MappingArray

    e.gen += 1
    if (e.gen == 1)
        Cambrian.fitness_evaluate!(e;fitness=evaluate)
        for i in eachindex(e.population)
            MAPElites.add_to_map(map_el,map_ind_to_b(MappingArray[i]),e.population[i],e.population[i].fitness)
        end
        MappingArray = []
    else
        new_pop = Array{Individual}(undef,0)

        # elites stay in population
        if e.cfg["n_elite"] > 0
            sort!(e.population)
            append!(new_pop,
                    e.population[(length(e.population)-e.cfg["n_elite"]+1):end])
        end

        # We are generating much more children than nb_population. It is in order to maximize population diversity
        # Indeed, with the help of the Dota_simulator, real evaluation is far too expensive, we simulate those
        # children behavior and select the nb_population individuals which are the further from each other. Then
        # we evaluate, with the actual game, those individuals and add them to the map.
        huge_new_pop = Array{Individual}(undef,0)
        # a big offsprings generation append here
        for i in 1:100
            p1 = MAPElites.select_random(map_el)
            child = deepcopy(p1)

            if e.cfg["p_crossover"] > 0 && rand() < e.cfg["p_crossover"]
                parents = vcat(p1, select_random(map_el))
                child = crossover(parents...)
            end

            if e.cfg["m_rate"] > 0 && rand() < e.cfg["m_rate"]
                child = mutation(child)
            end

            push!(huge_new_pop, child)
        end

        # adapt so keep_top+nb_elite=n_population
        keep_top = e.cfg["n_population"] - e.cfg["n_elite"]
        if e.cfg["python"]
            selected_ind = select_diverse(huge_new_pop,keep_top)
        else
            selected_ind = select_random(huge_new_pop, keep_top)
        end
        # we add them to the elites for next generation
        for ind in selected_ind
            push!(new_pop,ind)
        end

        e.population = new_pop
        Cambrian.fitness_evaluate!(e;fitness=evaluate)
        # once invidual are evaluated we can add them to the Map
        for i in eachindex(e.population)
            MAPElites.add_to_map(map_el,map_ind_to_b(MappingArray[i]),e.population[i],e.population[i].fitness)
        end

        MappingArray = []
    end

    if ((e.cfg["log_gen"] > 0) && mod(e.gen, e.cfg["log_gen"]) == 0)
        Cambrian.log_gen(e)
    end

    if ((e.cfg["save_gen"] > 0) && mod(e.gen, e.cfg["save_gen"]) == 0)
        Cambrian.save_gen(e)
        mapPath = "map/$(e.id)/$(e.gen)"
        save_map(map_el,mapPath)
    end
end

function MapelitesDotaRun!(e::Evolution,
                        mapel :: MAPElites.MapElites,
                        map_ind_to_b::Function;
                        mutation::Function=Cambrian.uniform_mutation,
                        crossover::Function=Cambrian.uniform_crossover,
                        evaluate::Function=Cambrian.random_evaluate)

    for i in (e.gen+1):e.cfg["n_gen"]
        MapelitesDotaStep!(e,mapel,map_ind_to_b;
                        mutation = mutation,
                        crossover = crossover,
                        evaluate = evaluate)
    end
end

"""
This function make an Individual play a single game of DOTA2

Once the game is finished it is adding behavior variables, used to get the
individual coordinates in the behavor space, to the MappingArray.

Return the fitness
"""
function EvaluateMapElites(ind::Individual)
    global MappingArray
    global totalDamageToOpp
    global ratioDamageTowerOpp
    # define the fitness function
    fitness = PlayDota(ind)
    push!(MappingArray,[totalDamageToOpp,ratioDamageTowerOpp])
    fitness
end
