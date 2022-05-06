strategy s = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("./strategies/large_strategy.json")
simulate 1 [<=300] {Ball(0).p, Ball(0).v, LearnerPlayer.fired} under s
