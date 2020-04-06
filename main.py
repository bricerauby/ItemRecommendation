
import snap

graph = snap.LoadEdgeList(snap.PUNGraph,'data/amazon-meta.txt')



CntV = snap.TIntPrV()

snap.GetWccSzCnt(graph, CntV)
for p in CntV:
    print("size %d: count %d" % (p.GetVal1(), p.GetVal2()))

snap.GetOutDegCnt(graph, CntV)
for p in CntV:
    print("degree %d: count %d" % (p.GetVal1(), p.GetVal2()))