import thirdeye

if __name__ == "__main__":
    ti = thirdeye.Thirdeye(force_t=True, max_for_class=14600, network='odin_v2')
    # ti.set_network('providence_v2')
    # ti.set_network('odin_v1')
    # ti.set_network('odin_v2')
    # ti.set_network('horus')

    # ti.evaluate()
    # ti.classify()
    #
    # ti.set_network('horus')
    # ti.train()
    # ti.perform_preprocessing()


    # thirdeye2 = thirdeye.Thirdeye(pre_p=False, force_t=False, name='odin_v1', evaluate=True)
    # thirdeye3 = thirdeye.Thirdeye(pre_p=False, force_t=False, name='horus', evaluate=True)
