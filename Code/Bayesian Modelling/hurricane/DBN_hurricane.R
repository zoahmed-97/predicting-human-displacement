#load libraries 
library(bnlearn)
library(caret)
library(e1071)
library(pROC)
library(bnviewer)
library(lattice)
set.seed(123)

#load the data
hurricane_df <- read.csv('/Users/zo/Desktop/SDS_Mobility/Code/Bayesian Modelling/hurricane/all_timesteps_hurricane.csv')
hurricane_df$X<- NULL

colSums(is.na(hurricane_df))

#explore the data
head(hurricane_df)
names(hurricane_df)

#create a blacklist for structure learning of the DBN 
#edges cannot go from a timestep to a previous timestep
bl1 <- tiers2blacklist(list(names(hurricane_df)[1:12], names(hurricane_df)[13:45]))
bl2 <- tiers2blacklist(list(names(hurricane_df)[13:23], names(hurricane_df)[24:45]))
bl3 <- tiers2blacklist(list(names(hurricane_df)[24:34], names(hurricane_df)[35:45]))
bl4 <- ordering2blacklist(c('t1_dist_to_hl', 't2_dist_to_hl', 't3_dist_to_hl'))
bl5 <- tiers2blacklist(list(names(hurricane_df)[13:22], 't1_dist_to_hl'))
bl5 <- tiers2blacklist(list(names(hurricane_df)[24:33], 't2_dist_to_hl'))
bl6 <- tiers2blacklist(list(names(hurricane_df)[35:44], 't3_dist_to_hl'))
bl_hurricane <- rbind(bl1, bl2, bl3, bl4, bl5, bl6)
bl_hurricane <- unique(bl_hurricane)
wl = data.frame(from = c("t0_rog", "t0_entropy_step"), to = c("t1_dist_to_hl", "t1_dist_to_hl"))
#structure learning with different structure learning algorithms
#first with score-based (hill climbing)
dyn.dag.hc <- hc(hurricane_df, blacklist=bl_hurricane, whitelist=wl)
dyn.dag.tabu <- tabu(hurricane_df, blacklist=bl_hurricane, whitelist=wl)
#then with hybrid (min-max hill climbing)
dyn.dag.mmhc <- mmhc(hurricane_df, blacklist=bl_hurricane, whitelist=wl)

#now compare the scores
score(dyn.dag.hc, data=hurricane_df)
score(dyn.dag.tabu, data=hurricane_df)
score(dyn.dag.mmhc, data=hurricane_df)
#tabu perfoms best

#model averaging to ensure greater confidence in edges and their directions
#done for tabu only as it was the best model
str.diff.hurricane <- boot.strength(hurricane_df, R = 200, 
                                    algorithm = "tabu",
                                    algorithm.args = list(blacklist = bl_hurricane, 
                                                whitelist= wl))
#inspecting arc strenghts, directions and threshold that will be used to decide whether an arc is strong enough to be included in the consensus network
attr(str.diff.hurricane, "threshold") #threshold is 0.48
str.diff.hurricane[which(str.diff.hurricane$to == 't1_dist_to_hl' & str.diff.hurricane$strength >= 0.48),]
str.diff.hurricane[which(str.diff.hurricane$to == 't2_dist_to_hl' & str.diff.hurricane$strength >= 0.48),]
str.diff.hurricane[which(str.diff.hurricane$to == 't3_dist_to_hl' & str.diff.hurricane$strength >= 0.48),]
#averaging
dyn.dag.tabu <- averaged.network(str.diff.hurricane)
undirected.arcs(dyn.dag.tabu)

#---ignore---
#one arc is undirected, checking scores for the different directions the arc can take
#choose.direction(dyn.dag.tabu, data = hurricane_df, c("t0_rain_intensity", "t0_dist2_urbancentres"),
#                 criterion = "aic", debug= TRUE)
#dyn.dag.tabu <- drop.arc(dyn.dag.tabu, from = "t0_rain_intensity", to = "t0_dist2_urbancentres")
#---ignore---

#first test how well the network structure learnt performs against a random network structure
#initialise a random network
rand <- random.graph(names(hurricane_df))
score(rand, data=hurricane_df, type='bic')
score(dyn.dag.tabu, data=hurricane_df, type='bic')


#second, test how well the parameter learning worked by computing the predictive error
#the target node here is the distance from home location (at three time periods)
val1 <-  bn.cv(hurricane_df, dyn.dag.tabu, loss='pred-lw', 
               loss.args = list(target='t1_dist_to_hl'), 
               fit='bayes', fit.args = list(iss = 100))
val1
OBS1 = unlist(lapply(val1, `[[`, "observed"))
PRED1 = unlist(lapply(val1, `[[`, "predicted"))
results1 <- as.matrix(confusionMatrix(OBS1, PRED1), what="classes")
results1 <- round(results1, 2)
confusionMatrix(OBS1, PRED1)
write.csv(results1, file="hurricane_res_dist1.csv")

val2 <-  bn.cv(hurricane_df, dyn.dag.tabu, loss='pred-lw', loss.args = list(target='t2_dist_to_hl'),
               fit='bayes', fit.args = list(iss = 100))
val2
OBS2 = unlist(lapply(val2, `[[`, "observed"))
PRED2 = unlist(lapply(val2, `[[`, "predicted"))
results2 <- as.matrix(confusionMatrix(OBS2, PRED2), what="classes")
results2 <- round(results2, 2)
confusionMatrix(OBS2, PRED2)
write.csv(results2, file="hurricane_res_dist2.csv")


val3 <-  bn.cv(hurricane_df, dyn.dag.tabu, loss='pred-lw', 
               loss.args = list(target='t3_dist_to_hl'),
               fit='bayes', fit.args = list(iss = 100))
val3
OBS3 = unlist(lapply(val3, `[[`, "observed"))
PRED3 = unlist(lapply(val3, `[[`, "predicted"))
results3 <- as.matrix(confusionMatrix(OBS3, PRED3), what="classes")
results3 <- round(results3, 2)
confusionMatrix(OBS3, PRED3)
write.csv(results3, file="hurricane_res_dist3.csv")


#fit the network to the whole dataset and see the CPTs of the distance nodes
fitted <- bn.fit(dyn.dag.tabu, hurricane_df, method = "bayes")
fitted$t1_dist_to_hl
fitted$t2_dist_to_hl
fitted$t3_dist_to_hl



#plot the graph
t0_nodes = list(names(hurricane_df)[1:12])[[1]]
t1_nodes = list(names(hurricane_df)[13:22])[[1]]
t2_nodes = list(names(hurricane_df)[24:33])[[1]]
t3_nodes=  list(names(hurricane_df)[35:44])[[1]]
distance_nodes = list('t1_dist_to_hl', 't2_dist_to_hl', 't3_dist_to_hl')

#plot entire graph
viewer(dyn.dag.tabu,
       bayesianNetwork.width = "80%",
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
dist1_nodes_mb= (mb(dyn.dag.tabu, 't1_dist_to_hl'))
dist2_nodes_mb= (mb(dyn.dag.tabu, 't2_dist_to_hl'))
dist3_nodes_mb= (mb(dyn.dag.tabu, 't3_dist_to_hl'))
dist_nodes = c(dist1_nodes_mb, dist2_nodes_mb, dist3_nodes_mb)
dist_nodes = unique(dist_nodes)
dist_graph = subgraph(dyn.dag.tabu, dist_nodes)
nodes(dist_graph)

distance_color= c('t1_dist_to_hl', 't2_dist_to_hl', 't3_dist_to_hl')
non_distance_color = setdiff(dist_nodes, distance_color)

viewer(dist_graph,
       bayesianNetwork.width = "100%",
       bayesianNetwork.height = "100vh",
       bayesianNetwork.layout = "layout_nicely",
       edges.dashes = TRUE,
       node.colors = list(background = "white",
                          border = "black",
                          highlight = list(background = "#e91eba",
                                           border = "black")),
       node.font = list(color= 'black', size=20, weight='bold'),
       clusters.legend.options = list(
               list(label = "Other nodes",
                    shape = "icon",
                    icon = list(code = "f111",
                                size = 10,
                                color = "#7f9cd3")),
               list(label = "Distance nodes",
                    shape = "icon",
                    icon = list(code = "f111",
                                size = 10,
                                color = "#f3514c"))),
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


