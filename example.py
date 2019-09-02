import thirdeye
"""
An example of the Thirdeye system used to classify unknown videos. 
"""
if __name__ == "__main__":
    ti = thirdeye.Thirdeye(max_for_class=14600, network='odin_v2')
    ti.classify()
    ti.set_network('providence_v2')
    ti.classify()
    ti.set_network('odin_v1')
    ti.classify()
    ti.set_network('odin_v2')
    ti.classify()
    ti.set_network('horus')
    ti.classify()
    # ti.evaluate()
