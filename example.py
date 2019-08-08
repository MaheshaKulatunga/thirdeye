import thirdeye

if __name__ == "__main__":
    ti = thirdeye.Thirdeye()
    ti.set_network('odin')
    # ti.evaluate()
    ti.classify()

    ti.set_network('horus')
    # ti.train()
    # ti.perform_preprocessing()


    # thirdeye2 = thirdeye.Thirdeye(pre_p=False, force_t=False, name='odin', evaluate=True)
    # thirdeye3 = thirdeye.Thirdeye(pre_p=False, force_t=False, name='horus', evaluate=True)
