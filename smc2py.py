import re

class SMCSimulation:


    def __init__(self, line, is_gui = False):
        self.points = []
        if(is_gui):
            self.loadFromGUI(line)
        else:
            self.loadFromEngine(line)



    def loadFromGUI(self, line):

        return


    def loadFromEngine(self, line):
        m = re.findall(r'\(\s*(-\d+|-\d+\.\d+|\d+|\d+\.\d+)\s*,\s*(-\d+|-\d+\.\d+|\d+|\d+\.\d+)\s*\)', line)
        for point in m:
            self.points.append((float(point[0]), float(point[1])))
        return

    def predict(self, time, hint = 0):

        value = 0

        lx = 0
        lval = 0

        found = False

        while hint < len(self.points):
            (x, val) = self.points[hint]
            if x > time:
                found = True
                hint -= 1
                if lx == time:
                    value = [lval]
                    while self.points[hint][0] == time:
                        value.insert(0, self.points[hint][1])
                        if hint == 0:
                            break
                        hint -= 1
                    break
                else:
                    r = (val - lval) / (x - lx)
                    value = ((time - lx)*r) + lval
                    break
            hint += 1
            lx = x
            lval = val

        (t, v) = self.points[len(self.points) - 1]
        if not found and t != time:
            print("Time is out of range for simulation " + str(time))
            value = v
            hint = len(self.points) - 1

        return value, hint

    def __len__(self):
        return len(self.points)

    def horizon(self):
        (t, v) = self.points[len(self.points) - 1]
        return t



class SMCData:

    def __init__(self):
        self.fields = []
        self.data = []


    def loadFromGUI(self, stream):



        return self

    def loadFromEngine(self, stream):

        lastData = 0
        for line in stream:
            if line[0] == "[":
                m = re.search(r'\[(\d+)\]', line)
                if int(m.groups()[0]) != len(self.data[lastData]):
                    print("ERROR: expected simulation no " + str(len(self.data[lastData])) + " got " + m.groups[0])
                dindex = line.index(":")
                self.data[lastData].append(SMCSimulation(line[dindex + 1:], False))
            else:
                e = line.strip()
                if not e:
                    break # whitespace
                else:
                    lastData = len(self.fields)
                    self.fields.append(line[:len(line) - 2])  # remove : at the end
                    self.data.append([]) # add empty

        return self

    def predict(self, time, simulation=0, hint=None):
        result = []

        if hint == None:
            hint = []

        for f in range(len(self.fields)):
            if f >= len(hint):
                hint.append(0)
            (val, h) = self.data[f][simulation].predict(time, hint[f])
            result.append(val)
            hint[f] = h
        return result, hint

    def __len__(self):
        if len(self.fields) == 0:
            return 0
        else:
            return len(self.data[0])

    def variables(self):
        return self.fields

    def horizon(self):
        m = 0
        for arr in self.data:
            for d in arr:
                m = max(m, d.horizon())

        return m


def parseEngineOutput(filename):
    res = []
    with open(filename) as stream:

        for line in stream:
            if "Formula is satisfied" in line:
                d = SMCData()
                d.loadFromEngine(stream)
                res.append(d)
    return res

