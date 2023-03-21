import demo_sympy as ds

group_name = "the lazy Susans"
def make_basic_bridge():
    """Write what I do here.

    Explain how I work here/ what you need for me to work correctly"""
    l = float(input("What is the length?"))
    s = 4
    h = 10
    return ds.make_bridge(l, s, h)

def make_shaped_bridge():
    """You can figure this one out, yes?"""
    help(ds.make_f_bridge())


def make_custom_bridge(length=8, segments=4, height=2.5, name_me1=0, name_me2=0, name_me3=0):
    """Rewrite me to give your team an advantage in designing bridges that have special constraints

    What if the moving load is much heavier?
    What if the bridge is a lot longer?
    What if the nodes have to be underneath the road?
    What if nodes can only go above the road?
    """
    tt = ds.Truss2()
    # build road
    for i in range(segments + 1):
        tt.add_node('road_joint_%i' % i, i * length / segments, 0)
    for i in range(1, segments + 1):
        tt.add_node('truss_node_%i' % i, (i - 0.5) * length / segments, height)
        tt.add_member('beam_%ia' % i, 'road_joint_%i' % (i - 1), 'truss_node_%i' % i)
        tt.add_member('beam_%ib' % i, 'road_joint_%i' % i, 'truss_node_%i' % i)
    for i in range(2, segments + 1):
        tt.add_member('beam_%ic' % i, 'truss_node_%i' % (i - 1), 'truss_node_%i' % i)
    for i in range(1, segments + 1):
        tt.add_member('road_section_%i' % i, 'road_joint_%i' % (i - 1), 'road_joint_%i' % i)
    tt.apply_support('road_joint_0', 'pinned')
    tt.apply_support('road_joint_%i' % segments, 'roller')
    return tt


if __name__ == "__main__":
    # you can make this interactive, and it will run from command line
    # python bridge_designs_template.py
    tt = make_basic_bridge()
    ds.truss_simulator(tt, group=group_name)
