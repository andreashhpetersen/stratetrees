//This file was generated from (Academic) UPPAAL 4.1.20-stratego-3 (branch better_strategy_handling commit f479ab27b4756f51064bde11b40f882e6f541303), February 2018

/*
can i just say p for all balls? 
*/
strategy HitWell = minE (LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) [<=120] {} -> {Ball(0).p, Ball(0).v, Ball(1).p, Ball(1).v, Ball(2).p, Ball(2).v} : <> time >= 120

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell
