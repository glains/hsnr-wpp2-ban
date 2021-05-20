import os.path

class ContourReader:
    def __init__(self):
        pass

    def _parseContour(self,text):
        areas = []
        lines = [s.strip() for s in text.split('\n')]
        for line in lines:
            if line.find("AllPoints") > -1:
                line = line[9:]
                parts1 = line.split(":")
                if len(parts1[1]) == 0:
                    continue
                points = parts1[1].split(",")
                points = points[0:len(points)-1]# Remove last
                tarr = []
                for p in points:
                    parr = p.split("|")
                    tarr.append([int(parr[0]),int(parr[1])])
                areas.append(tarr)
        return areas
    
    def parseFile(self,path):
        if os.path.isfile(path):
            f = open(path, "r")
            txt = f.read()
            f.close()
            return self._parseContour(txt)
        return None