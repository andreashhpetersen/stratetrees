//This file was generated from (Academic) UPPAAL 4.1.20-stratego-9 (rev. 67D95DBCE6B8B4ED), January 2022

/*

*/
simulate 1 [<=300] {Ball(0).p}

/*
can i just say p for all balls? 
*/
// strategy HitWell = minE (LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) [<=120] {} -> {Ball(0).p, Ball(0).v}: <> time >= 120

/*

*/
//NO_QUERY

/*

*/
strategy HitWell_original = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("/home/andreashhp/Documents/university/direc/stratetrees/uppaal_models/ball/strategies/large_strategy.json")

/*

*/
strategy HitWell_converted = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("/home/andreashhp/Documents/university/direc/stratetrees/uppaal_models/ball/strategies/large_converted_noPrune.json")

/*

*/
strategy HitWell_zeroPruned = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("/home/andreashhp/Documents/university/direc/stratetrees/uppaal_models/ball/strategies/large_converted_zeroPruned.json")

/*

*/
strategy HitWell_onePruned = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("/home/andreashhp/Documents/university/direc/stratetrees/uppaal_models/ball/strategies/large_converted_onePruned.json")

/*

*/
strategy HitWell_twoPruned = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("/home/andreashhp/Documents/university/direc/stratetrees/uppaal_models/ball/strategies/large_converted_twoPruned.json")

/*

*/
//NO_QUERY

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell_original

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell_converted

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell_zeroPruned

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell_onePruned

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell_twoPruned

/*

*/
//NO_QUERY

/*

*/

strategy HitWell_zeroPruned2 = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("/home/andreashhp/Documents/university/direc/stratetrees/uppaal_models/ball/strategies/large_converted_zeroPruned2.json")

/*

*/
strategy HitWell_onePruned2 = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("/home/andreashhp/Documents/university/direc/stratetrees/uppaal_models/ball/strategies/large_converted_onePruned.json2")

/*

*/
strategy HitWell_twoPruned2 = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("/home/andreashhp/Documents/university/direc/stratetrees/uppaal_models/ball/strategies/large_converted_twoPruned2.json")

/*

*/
//NO_QUERY

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell_zeroPruned2

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell_onePruned2

/*

*/
E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under HitWell_twoPruned2
/*

*/
//NO_QUERY

/*

*/
simulate 1 [<=30] {LearnerPlayer.fired, Ball(0).p} under HitWell
