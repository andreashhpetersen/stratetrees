#!/bin/python

import smc2py
import os
import sys



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <infile> <outfile>")
        exit()
    infile = sys.argv[1]
    outfile = sys.argv[2]
    # we just expect the control signal to be the first of the observed

    from_engine = smc2py.parseEngineOutput(infile)
    trajectories = from_engine[-1]
    with open(outfile, "w") as out:
        for i in range(len(trajectories)):
            hints = [0 for _ in trajectories.fields]
            for (t, v) in trajectories.data[0][i].points:
                if v > 0:
                    # the controller is in power
                    values = []
                    old_hints = [h for h in hints]
                    for d in range(1, len(trajectories.fields)):
                        (value, hint) = trajectories.data[d][i].predict(t,hints[d])
                        values.append(str(value))
                        hints[d] = hint

                    # for Peter: this shows the problem
                    # if float(values[1]) < 2:
                    #     import ipdb; ipdb.set_trace()

                    out.write(",".join(values))
                    out.write("\n")
