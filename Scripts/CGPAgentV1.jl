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
- getting features and returning action
- getting the nbKill, nbDeath and earlyPenalty values
- handle early stopping
- close the server when a game is over
"""
function ServerHandler(request::HTTP.Request)

    global lastFeatures
    global individual
    global breezyIp
    global breezyPort
    global server
    global nbKill
    global nbDeath
    global earlyPenalty

    path = HTTP.URIs.splitpath(request.target)
    println("Path is: $path")
    # path is either an array containing "update" or nothing so the following line means "if there is an update"
    if (size(path)[1] != 0)
        """
        Update route is called, game finished.
        """

        println("Game done.")
        cfg["n_game"] += 1
        content = GetContent(request)
        # println(content)
        if (content["status"] == "DONE")
            nbDeath = content["deaths"]
            nbKill = content["radiantKills"]
        elseif (content["status"] == "CANCELED") # i.e EarlyStop
            earlyPenalty = 1
        end

        println("Fitness: $(Fitness1(lastFeatures,nbKill,nbDeath,earlyPenalty))")
        """
        Since the Game is over we want to close the server
        """
        # closing the server generate an error, in order to keep the code running we use a try & catch
        try
            close(server)
        catch e
            return HTTP.post(404,JSON.json(Dict("socket closed"=>"0")))
        end

    else
        """
        Relay route is called, gives features from the game for the agent.
        """

        println("Received features.")
        # get data as json, then save to list
        content = GetContent(request)
        # features = JSON.json(content)
        # println(features)
        lastFeatures = content
        # you need this conversion to call process
        lastFeatures = convert(Array{Float64},lastFeatures)

        if EarlyStop(lastFeatures)
            """
            EarlyStop will stopped the current game by calling the upgrade route
            """

            println("Early Stop.")
            # we send to the Breezy server to call the update route
            stopUrl = "http://$breezyIp:$breezyPort/run/active"
            response = HTTP.delete(stopUrl)
            println(response)

        else
            """
            Agent code to determine action from features.
            """
            action = argmax(process(individual, lastFeatures))-1 # julia array start at 1 but breezy server is python so you need the "-1"
            println("Action made: $action")
            PostResponse(Dict("actionCode"=>action))
        end

    end
end

"""
This function allow us to play one game and to get the fitness score of one individual
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
    # initialize lastFeatures
    lastFeatures = [0.0]
    # will run the game until it is over, when it is over there is error because of the server closure
    try
        HTTP.serve(ServerHandler,args["agentIp"],parse(Int64,args["agentPort"]);server=server)
    # when there is the error we know the game is over and we can return the fitness
    catch e
        [Fitness1(lastFeatures,nbKill,nbDeath,earlyPenalty)]
    end
end

"""
This function is the one generating population
"""
function Populate(evo::Cambrian.Evolution)
    mutation = i::CGPInd->goldman_mutate(cfg, i)
    Cambrian.oneplus_populate!(evo; mutation=mutation, reset_expert=true)
end

"""
This function is the one that allow us to evaluate an individual
"""
function Evaluate(evo::Cambrian.Evolution)
    # define the fitness function
    fit = i::CGPInd->PlayDota(i)
    Cambrian.fitness_evaluate!(evo; fitness=fit)
    evo.text = Formatting.format("{1:e}", cfg["n_game"])
end
