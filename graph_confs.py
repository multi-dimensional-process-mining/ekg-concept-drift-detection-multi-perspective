#####################################################
############# EXPLANATION OF SETTINGS ###############
#####################################################

# SETTING FOR PREPROCESSING
# if no preprocessing is done, empty strings can be assigned
filename = {}           # name of the csv file to build the graph from
column_names = {}       # names of the columns in the csv file for [case, activity, timestamp, resource(, lifecycle)] (in this order)
separator = {}          # separator used in csv file
timestamp_format = {}   # format of the timestamps recorded in the csv file

# GRAPH SETTINGS
password = {}                   # password of neo4j database
entity_labels = {}              # labels used in the graph for: [[df_resource, node_resource], [df_case, node_case]]
action_lifecycle_labels = {}    # labels used in the graph for: [activity, lifecycle]
timestamp_label = {}            # label used for timestamp
case_attr_labels = {}           # labels used for process instance attributes

# SETTINGS FOR VISUALIZING:
name_data_set = {}          # name of the data set, used for configuring the node labels when visualizing subgraphs (only available for bpic2014 and bpic2017)


#####################################################
############ CONFIGURATION OF SETTINGS ##############
#####################################################

# -------------- BPIC 2017 SETTINGS -----------------

for graph in ["bpic2017_susp_res"]:

    filename[graph] = "bpic2017"
    name_data_set[graph] = "bpic2017"
    column_names[graph] = ["case", "event", "time", "org:resource", "lifecycle:transition"]
    separator[graph] = ","
    timestamp_format[graph] = "%Y/%m/%d %H:%M:%S.%f"
    password[graph] = "bpic2017"

    entity_labels[graph] = [['resource', 'resource'],
                            ['case', 'case']]
    action_lifecycle_labels[graph] = ['activity', 'lifecycle']
    timestamp_label[graph] = "timestamp"
    case_attr_labels[graph] = ["ApplicationType", "LoanGoal", "RequestedAmount", "OfferID"]
    if graph in ["bpic2017_susp_res"]:
        column_names[graph] += case_attr_labels[graph]
