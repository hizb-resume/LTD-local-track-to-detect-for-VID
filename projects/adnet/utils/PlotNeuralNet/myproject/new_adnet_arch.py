import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
#input
    to_input( 'input.png' ),
    to_Conv("conv1", 3, 96, offset="(0,0,0)", to="(0,0,0)", height=53, depth=64, width=53),
    to_Conv("conv2", 96, 256, offset="(0,0,0)", to="(0,0,0)", height=53, depth=64, width=53 ),
    # to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv3", 256, 512, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    # to_connection( "pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    to_connection("pool2", "soft1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()