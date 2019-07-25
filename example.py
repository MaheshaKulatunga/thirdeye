import thirdeye

if __name__ == "__main__":
    ti = thirdeye.Thirdeye()
    ti.perform_preprocessing()
    ti.set_network('odin')
    ti.train()
    ti.evaluate()
    # ti.classify()

    # thirdeye2 = thirdeye.Thirdeye(pre_p=False, force_t=False, name='odin', evaluate=True)
    # thirdeye3 = thirdeye.Thirdeye(pre_p=False, force_t=False, name='horus', evaluate=True)
