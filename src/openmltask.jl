module OpenMLReaders

# based on MLJ OpenMLData module

using HTTP
using JSON

using DataFrames
using Random
using AutoMLPipeline.Utils
using AutoMLPipeline.AbsTypes

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!

export OpenMLTaskReader
export test_opmltaskreader

const OPENML_URL = "https://www.openml.org/api/v1/json"

mutable struct OpenMLTaskReader <: Transformer
	name::String
	model::Dict
	args::Dict

	function OpenMLTaskReader(args::Dict=Dict())
		default_args = Dict(
			:name => "openmltask",
			:task_id => 3
		)
		cargs=nested_dict_merge(default_args,args)
		cargs[:name] = cargs[:name]*"_"*string(cargs[:task_id])
		new(cargs[:name],Dict(),cargs)
	end
end

function OpenMLTaskReader(id::Int)
	OpenMLTaskReader(Dict(:task_id => id))
end

function fit!(opmlrdr::OpenMLTaskReader,x::DataFrame=DataFrame(), y::Vector=[])

  desc = load_task_description(opmlrdr)
  args = opmlrdr.args
  data_set_id = desc["task"]["input"][1]["data_set"]["data_set_id"]
  args[:data_set_id] = match(r"\d+",data_set_id).match |> x->parse(Int,x)
  opmlrdr.model = args
end

function transform!(opmlrdr::OpenMLTaskReader,x::DataFrame=DataFrame())
  id = opmlrdr.model[:data_set_id]
  rowtable =  load(id) 
  return rowtable |> DataFrame
end

function load(id::Int)
  response = load_Dataset_Description(id)
  arff_file = HTTP.request("GET", response["data_set_description"]["url"])
  return convert_ARFF_to_rowtable(arff_file)
end

function load_task_description(opmlrdr::OpenMLTaskReader)
	id = opmlrdr.args[:task_id]
	url = string(OPENML_URL,"/task/$id")
	try
		r = HTTP.request("GET", url)
		if r.status == 200
			return JSON.parse(String(r.body))
		elseif r.status == 110
			println("Please provide data_id.")
		elseif r.status == 111
			println("Unknown dataset. Data set description with data_id was not found in the database.")
		elseif r.status == 112
			println("No access granted. This dataset is not shared with you.")
		end
	catch e
		println("Error occurred : $e")
		return DataFrame()
	end
	return DataFrame()
end

"""
Returns a Vector of NamedTuples.
Receives an `HTTP.Message.response` that has an
ARFF file format in the `body` of the `Message`.
"""
function convert_ARFF_to_rowtable(response)
	data = String(response.body)
	data2 = split(data, "\n")
	featureNames = String[]
	dataTypes = String[]
	# TODO: make this more performant by anticipating types?
	named_tuples = [] # `Any` type here bad
	for line in data2
		if length(line) > 0
			if line[1:1] != "%"
				d = []
				if occursin("@attribute", lowercase(line))
					push!(featureNames, replace(split(line, " ")[2], "'" => ""))
					push!(dataTypes, split(line, " ")[3])
				elseif occursin("@relation", lowercase(line))
					nothing
				elseif occursin("@data", lowercase(line))
					# it means the data starts
					nothing
				else
					values = split(line, ",")
					for i in eachindex(featureNames)
						if lowercase(dataTypes[i]) in ["real","numeric"]
							push!(d, featureNames[i] => Meta.parse(values[i]))
						else
							# all the rest will be considered as String
							push!(d, featureNames[i] => values[i])
						end
					end
					push!(named_tuples, (; (Symbol(k) => v for (k,v) in d)...))
				end
			end
		end
	end
	return identity.(named_tuples) # not performant; see above
end

function load_Data_Features(id::Int; api_key::String = "")
	if api_key == ""
		url = string(OPENML_URL, "/data/features/$id")
	end
	try
		r = HTTP.request("GET", url)
		if r.status == 200
			return JSON.parse(String(r.body))
		elseif r.status == 271
			println("Unknown dataset. Data set with the given data ID was not found (or is not shared with you).")
		elseif r.status == 272
			println("No features found. The dataset did not contain any features, or we could not extract them.")
		elseif r.status == 273
			println("Dataset not processed yet. The dataset was not processed yet, features are not yet available. Please wait for a few minutes.")
		elseif r.status == 274
			println("Dataset processed with error. The feature extractor has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, please contact the API admins.")
		end
	catch e
		println("Error occurred : $e")
		return nothing
	end
	return nothing
end

"""
Returns information about a dataset. The information includes the name,
information about the creator, URL to download it and more.
- 110 - Please provide data_id.
- 111 - Unknown dataset. Data set description with data_id was not found in the database.
- 112 - No access granted. This dataset is not shared with you.
"""
function load_Dataset_Description(id::Int; api_key::String="")
    url = string(OPENML_URL, "/data/$id")
    try
        r = HTTP.request("GET", url)
        if r.status == 200
            return JSON.parse(String(r.body))
        elseif r.status == 110
            println("Please provide data_id.")
        elseif r.status == 111
            println("Unknown dataset. Data set description with data_id was not found in the database.")
        elseif r.status == 112
            println("No access granted. This dataset is not shared with you.")
        end
    catch e
        println("Error occurred : $e")
        return nothing
    end
    return nothing
end


function test_opmltaskreader()
  tasks = [3021,3022,3481,3549,3560,3573,3902,3903,
			  3904,3913,3917,3918,7592,9910,9946,9952,9957,9960,
			  9964,9971,9976,9977,9978,9981,9985,10093,10101,14952,
			  14954,14965,14969,14970,125920,125922,146195,146800,146817,146819,
			  146820,146821,146822,146824,146825,167119,167120,167121,167124,167125,
			  167140,167141,14,15,2079,43,29,3,6,11,12,16,18,22,23,
			  28,31,32,37,45,49,53,219,
			  2074,]
  for task in tasks
	 try
		treader = OpenMLTaskReader(task)
		df = fit_transform!(treader)
		println(task,":",size(df))
	 catch e
		println(e)
		println("error:",task)
	 end
  end
end



end
