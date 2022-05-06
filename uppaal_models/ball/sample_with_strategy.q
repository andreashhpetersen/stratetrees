strategy s = loadStrategy {} -> {Ball(0).p, Ball(0).v} ("../peter_python/animation/ball_strategy")
simulate 1 [<=300] {Ball(0).p, Ball(0).v, LearnerPlayer.fired} under s
