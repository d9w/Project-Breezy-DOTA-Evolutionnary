using ArgParse
using PyCall

s = ArgParseSettings()
@add_arg_table! s begin
    "--simulator"
    help = "dota simulator path"
    default = "C:\\Users\\denni\\Documents\\GitHub\\Dota_Simulator"
end

args = parse_args(ARGS, s)
pushfirst!(PyVector(pyimport("sys")."path"), args["simulator"])
dotasimlib = pyimport("DOTA_simulator")
