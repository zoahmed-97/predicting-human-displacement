#load libraries 
library(bnlearn)
library(caret)
library(e1071)
library(pROC)
library(bnviewer)
set.seed(123)
#load the data
earthquake_df <- read.csv('/Users/zo/Desktop/SDS_Mobility/Code/Bayesian Modelling/earthquake/all_timesteps_earthquake.csv')
earthquake_df$X<- NULL

#explore the data
head(earthquake_df)
names(earthquake_df)

#create a blacklist for structure learning of the DBN 
#edges cannot go from a timestep to a previous timestep
bl1_eq <- tiers2blacklist(list(names(earthquake_df)[1:8], names(earthquake_df)[9:29]))
bl2_eq <- tiers2blacklist(list(names(earthquake_df)[9:15], names(earthquake_df)[16:29]))
bl3_eq<- tiers2blacklist(list(names(earthquake_df)[16:22], names(earthquake_df)[23:29]))
bl4_eq <- ordering2blacklist(c('t1_dist_to_hl', 't2_dist_to_hl', 't3_dist_to_hl'))
bl_eq <- rbind(bl1_eq, bl2_eq, bl3_eq, bl4_eq)
bl_eq <- unique(bl_eq)

#structure learning with different structure learning algorithms
#first with score-based (hill climbing)
dyn.dag.hc.eq <- hc(earthquake_df, blacklist=bl_eq, score='bde')
dyn.dag.tabu.eq <- tabu(earthquake_df, blacklist=bl_eq, score='bde')
#then with hybrid (min-max hill climbing)
dyn.dag.mmhc.eq <- mmhc(earthquake_df, blacklist=bl_eq, score='bde')

#now compare the scores
score(dyn.dag.hc.eq, data=earthquake_df, type='bic')
score(dyn.dag.tabu.eq, data=earthquake_df, type='bic')
score(dyn.dag.mmhc.eq, data=earthquake_df, type='bic')
#tabu perfoms best

#model averaging to ensure greater confidence in edges and their directions
#done for tabu only as it was the best model
str.diff <- boot.strength(earthquake_df, R = 200, algorithm = "tabu",
                         algorithm.args = list(blacklist = bl_eq))
#inspecting arc strenghts, directions and threshold that will be used to decide whether an arc is strong enough to be included in the consensus network
attr(str.diff, "threshold") #threshold is 0.495
str.diff[which(str.diff$to == 't1_dist_to_hl' & str.diff$strength >= 0.495),]
str.diff[which(str.diff$to == 't2_dist_to_hl' & str.diff$strength >= 0.495),]
str.diff[which(str.diff$to == 't3_dist_to_hl' & str.diff$strength >= 0.495),]
#averaging
dyn.dag.tabu.eq <- averaged.network(str.diff)
score(dyn.dag.tabu.eq, data=earthquake_df, type='bic')

#first test how well the network structure learnt performs against a random network structure
#initialise a random network
rand_eq <- random.graph(names(earthquake_df))

score(rand_eq, data=earthquake_df, type='bic')
score(dyn.dag.tabu.eq, data=earthquake_df, type='bic')

#second, test how well the parameter learning worked by computing the predictive error
#the target node here is the distance from home location (at three time periods)
val1 <-  bn.cv(earthquake_df, dyn.dag.tabu.eq, loss='pred-lw', 
               loss.args = list(target='t1_dist_to_hl'),
               fit='bayes', fit.args = list(iss = 1))
val1
OBS1 = unlist(lapply(val1, `[[`, "observed"))
PRED1 = unlist(lapply(val1, `[[`, "predicted"))
results1 <- as.matrix(confusionMatrix(OBS1, PRED1), what="classes")
results1 <- round(results1, 2)
confusionMatrix(OBS1, PRED1)
write.csv(results1, file="eq_res_dist1.csv")

val2 <-  bn.cv(earthquake_df, dyn.dag.tabu.eq, loss='pred-lw', 
               loss.args = list(target='t2_dist_to_hl'),
               fit='bayes', fit.args = list(iss = 1))
val2
OBS2 = unlist(lapply(val2, `[[`, "observed"))
PRED2 = unlist(lapply(val2, `[[`, "predicted"))
results2 <- as.matrix(confusionMatrix(OBS2, PRED2), what="classes")
results2 <- round(results2, 2)
confusionMatrix(OBS2, PRED2)
write.csv(results2, file="eq_res_dist2.csv")

val3 <-  bn.cv(earthquake_df, dyn.dag.tabu.eq, loss='pred-lw', 
               loss.args = list(target='t3_dist_to_hl'),
               fit='bayes', fit.args = list(iss = 1))
val3
OBS3 = unlist(lapply(val3, `[[`, "observed"))
PRED3 = unlist(lapply(val3, `[[`, "predicted"))
results3 <- as.matrix(confusionMatrix(OBS3, PRED3), what="classes")
results3 <- round(results3, 2)
confusionMatrix(OBS3, PRED3)

write.csv(results3, file="eq_res_dist3.csv")

#fit the network to the whole dataset and see the CPTs of the distance nodes
fitted <- bn.fit(dyn.dag.tabu.eq, earthquake_df, method = "bayes")
fitted$t1_dist_to_hl
fitted$t2_dist_to_hl
fitted$t3_dist_to_hl

#plot the graph
names(earthquake_df)
t0_nodes = list(names(earthquake_df)[1:8])[[1]]
t1_nodes = list(names(earthquake_df)[9:14])[[1]]
t2_nodes = list(names(earthquake_df)[16:21])[[1]]
t3_nodes=  list(names(earthquake_df)[23:28])[[1]]
distance_nodes = list('t1_dist_to_hl', 't2_dist_to_hl', 't3_dist_to_hl')

#plot entire graph
viewer(dyn.dag.tabu.eq,
       bayesianNetwork.width = "100%",
       bayesianNetwork.height = "250vh",
       bayesianNetwork.layout = "layout_components",
       edges.dashes = TRUE,
       node.colors = list(background = "white",
                          border = "black",
                          highlight = list(background = "#e91eba",
                                           border = "black")),
       node.font = list(color= 'black', size=20, weight='bold'),
       #clusters.legend.title = list(text = "<b>Legend</b> <br> Variable type and timestep",
       #                             style = "font-size:12px;
       #                             font-family:Arial;
       #                             color:black;
       #                             font-weight: bold;
       #                             text-align:center;"),
       clusters.legend.options = list(
         list(label = "Time at disaster",
              shape = "icon",
              icon = list(code = "f111",
                          size = 10,
                          color = "#7f9cd3")),
         list(label = "1 week post disaster",
              shape = "icon",
              icon = list(code = "f111",
                          size = 10,
                          color = "#50a7ef")),
         list(label = "2-5 weeks post disaster",
              shape = "icon",
              icon = list(code = "f111",
                          size = 10,
                          color = "#5dcfe2")),
         list(label = ">5 weeks post disaster",
              shape = "icon",
              icon = list(code = "f111",
                          size = 10,
                          color = "#d19ce2")), 
         list(label = "Displacement variables",
              shape = "icon",
              icon = list(code = "f0c8",
                          size = 10,
                          color = "#f3514c"))
       ),
       clusters = list(
         list(label = "Variables for time of disaster",
              shape = "icon",
              icon = list(code = "f111", color = "#7f9cd3"),
              nodes = t0_nodes),
         list(label = "1 week post disaster",
              shape = "icon",
              icon = list(code = "f111", color = "#50a7ef"),
              nodes = t1_nodes),
         list(label = "2-5 weeks post disaster",
              shape = "icon",
              icon = list(code = "f111", color = "#5dcfe2"),
              nodes = t2_nodes),
         list(label = ">5 weeks post disaster",
              shape = "icon",
              icon = list(code = "f111", color = "#d19ce2"),
              nodes = t3_nodes),
         list(label = "Displacement variables",
              shape = "icon",
              icon = list(code = "f0c8", color = "#f3514c"),
              nodes = distance_nodes)
       ))

#plot subgraph of only distance nodes and their Markov Blankets
dist1_nodes_mb= (mb(dyn.dag.tabu.eq, 't1_dist_to_hl'))
dist2_nodes_mb= (mb(dyn.dag.tabu.eq, 't2_dist_to_hl'))
dist3_nodes_mb= (mb(dyn.dag.tabu.eq, 't3_dist_to_hl'))
dist_nodes = c(dist1_nodes_mb, dist2_nodes_mb, dist3_nodes_mb)
dist_nodes = unique(dist_nodes)
dist_graph = subgraph(dyn.dag.tabu.eq, dist_nodes)
nodes(dist_graph)

distance_color= c('t1_dist_to_hl', 't2_dist_to_hl', 't3_dist_to_hl')
non_distance_color = setdiff(dist_nodes, distance_color)

viewer(dist_graph,
       bayesianNetwork.width = "100%",
       bayesianNetwork.height = "120vh",
       bayesianNetwork.layout = "layout_components",
       edges.dashes = TRUE,
       node.colors = list(background = "white",
                          border = "black",
                          highlight = list(background = "#e91eba",
                                           border = "black")),
       node.font = list(color= 'black', size=16, weight='bold'),
       clusters = list(
         list(label = "Other nodes",
              shape = "icon",
              icon = list(code = "f111", color = "#7f9cd3"),
              nodes = non_distance_color),
         list(label = "1 week post disaster",
              shape = "Distance nodes",
              icon = list(code = "f111", color = "#f3514c"),
              nodes = distance_color))
)

#test how structure learnt for hurricane data perform on earthquake data
testdf_t2 <- subset(earthquake_df, select = c(t2_dist_to_hl, t1_dist_to_hl, t2_known_locs))
wl_t2 <- data.frame(from = c("t1_dist_to_hl", "t2_known_locs"), 
                                to = c("t2_dist_to_hl", "t2_dist_to_hl"))

dyn.dag.tabu.testdf2 <- tabu(testdf_t2, whitelist=wl_t2, score='bde')

val.test_t2 <-  bn.cv(testdf_t2, dyn.dag.tabu.testdf2, loss='pred-lw', 
               loss.args = list(target='t2_dist_to_hl'),
               fit='bayes', fit.args = list(iss = 1))
val.test_t2
OBS_test2 = unlist(lapply(val.test_t2, `[[`, "observed"))
PRED_test2 = unlist(lapply(val.test_t2, `[[`, "predicted"))
results_test2 <- as.matrix(confusionMatrix(OBS_test2, PRED_test2), what="classes")
results_test2 <- round(results_test2, 2)
confusionMatrix(OBS_test2, PRED_test2)
confusionMatrix(OBS2, PRED2)


