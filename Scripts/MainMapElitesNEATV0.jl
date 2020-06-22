using HTTP
using Random
using JSON
using Cambrian
using ArgParse
using Sockets
using Formatting
using Dates
using PyCall
using NEAT

include("../MAPElites/src/MapElites.jl")
include("../Scripts/Utils.jl")
include("../Scripts/surrogate.jl")
include("../Scripts/AgentMapElitesNEATModele.jl")

# SETTINGS
s = ArgParseSettings()
@add_arg_table! s begin
    "--breezyIp"
    help = "breezy server IP adress"
    default = "127.0.0.1"
    "--breezyPort"
    help = "breezy server port number"
    default = "8085"
    "--agentIp"
    help = "agent server IP adress"
    default = "127.0.0.1"
    "--agentPort"
    help = "agent server port number"
    default = "8086"
    "--startData"
    help = "the initial number of games launch when the agent is started"
    arg_type = Dict{String}{Any}
    default = Dict(
                "agent"=> "MapElitesNEATAgent",
                "size"=> 1
            )
    "--cfg"
    help = "configuration script"
    default = "Config/MapElitesNEATAgent.yaml"
    "--gen"
    help = "load existing generation"
    default = ""
    "--map"
    help = "load map"
    default = ""
    "--simulator"
    help = "dota simulator path"
    default = "C:\\Users\\denni\\Documents\\GitHub\\Dota_Simulator"
    "--python"
    help = "use python"
    action = :store_true
end

args = parse_args(ARGS, s)
cfg = get_config(args["cfg"])
cfg["python"] = args["python"]

if args["python"]
  pushfirst!(PyVector(pyimport("sys")."path"), args["simulator"])
  include("../Scripts/Julia_interface.jl")
end

# add to cfg the number of input(i.e nb of feature) and output
cfg["n_in"] = 113
cfg["n_out"] = 30

cfg["n_game"] = 0

# add to cfg the cfg of MapElites
cfg["features_dim"] = 2
cfg["grid_mesh"] = 50

"""
Declare variables global that you want the agent server to have access to.
"""
global breezyIp
global breezyPort
global agentIp
global agentPort
global startData
global oldLastFeatures
global lastFeatures
global server
global individual
global nbKill
global nbDeath
global earlyPenalty
global totalDamageToOpp
global ratioDamageTowerOpp
global MappingArray

breezyIp = args["breezyIp"]
breezyPort = args["breezyPort"]
agentIp = args["agentIp"]
agentPort = args["agentPort"]
startData = args["startData"]
# to be able to evaluate the fitness
lastFeatures = [0.0]
oldLastFeatures = [0.0]
# the server will be reinitialize when playing Dota
server = "whatever"
# the individual will be properly set when calling PlayDota(ind)
individual = "not_initialized"
# initialize variables of the fitness function
nbKill = 0
nbDeath = 0
earlyPenalty = 0
# initialize variables of the characterization function
totalDamageToOpp = 0
ratioDamageTowerOpp = 0
MappingArray = []
# initialize MapElites parameters
featuresDim = cfg["features_dim"]
gridMesh = cfg["grid_mesh"]
# define the mutation
mutation = i::NEATInd->mutate(i, cfg)

"""
MAIN LOOP
"""

e = Cambrian.Evolution(NEATInd, cfg; id = Dates.format(Dates.now(), "dd-mm-yyyy-HH-MM"))
if args["gen"] != ""
    LoadGen(e, args["gen"])
end
mapel = MAPElites.MapElites(featuresDim,gridMesh)
if args["map"] != ""
    mapel = load_map(args["map"])
    e.gen += 1
end
MapelitesDotaRun!(e,mapel,MapIndToB;mutation=mutation,evaluate=EvaluateMapElites)

best = sort(e.population)[end]
println("Final fitness: ", best.fitness[1])
