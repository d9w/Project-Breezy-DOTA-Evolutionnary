using HTTP
using Random
using JSON
using CartesianGeneticProgramming
using Cambrian
using ArgParse
using Sockets
using Formatting
using Dates
using PyCall

include("../MAPElites/src/MapElites.jl")
include("../Scripts/Utils.jl")
include("../Scripts/AgentMapElitesCGPModele.jl")

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
                "agent"=> "MapElitesCGPAgent",
                "size"=> 1
            )
    "--cfg"
    help = "configuration script"
    default = "Config/MapElitesCGPAgent.yaml"
    "--gen"
    help = "load existing generation"
    default = ""
    "--map"
    help = "load map"
    default = ""
end

args = parse_args(ARGS, s)
cfg = get_config(args["cfg"])

path_to_dota_simulator = args["simulator"]
pushfirst!(PyVector(pyimport("sys")."path"), path_to_Dota_Simulator)
include("../Scripts/Julia_interface.jl")

# add to cfg the number of input(i.e nb of feature) and output
cfg["n_in"] = 310
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
mutation = i::CGPInd->goldman_mutate(cfg, i)

"""
MAIN LOOP
"""

e = Cambrian.Evolution(CGPInd, cfg; id = Dates.format(Dates.now(), "dd-mm-yyyy-HH-MM"))
if args["gen"] != ""
    LoadGen(e, args["gen"])
end
mapel = MAPElites.MapElites(featuresDim,gridMesh)
if args["map"] != ""
    LoadGen(e, args["map"])
end
MapelitesDotaRun!(e,mapel,MapIndToB;mutation=mutation,evaluate=EvaluateMapElites)

best = sort(e.population)[end]
println("Final fitness: ", best.fitness[1])
