#!/usr/bin/env python
#=========================================================================
# USAGE: python csr2csc.py [options]
#=========================================================================
# convert .npz SCR format sparse matrices into CSC format
#
#     -h --help                     Display this message
#     -d --distribution     Select the distribution of the matrix
#                                                 <uniform> : uniform distribution (default)
#                                                 <reddit>    : reddit matrix
#                                                 <> : -- other distributions are currently not supported --
#     -s --size                     Choose size of the matrix (only work for uniform distribution)
#                                                 <all>    : all other options (default)
#                                                 <10K>    : 10K x 10K
#                                                 <100K> : 100K x 100K
#     -g --degree                 Choose the average degree of the graph (only work for uniform distribution)
#                                                 <all>    : all other options (default)
#                                                 <10>
#                                                 <100>
#                                                 <1000>
#     -t --type                     Choose the data type
#                                                 <all>    : all other options (default)
#                                                 <int>
#                                                 <float>
#
#

import os
import sys
import argparse
import scipy.sparse

#-------------------------------------------------------------------------
# Command line processing
#-------------------------------------------------------------------------

class ArgumentParserWithCustomError(argparse.ArgumentParser):
    def error( self, msg = "" ):
        if ( msg ): print("\n ERROR: %s" % msg)
        print("")
        file = open( sys.argv[0] )
        for ( lineno, line ) in enumerate( file ):
            if ( line[0] != '#' ): sys.exit(msg != "")
            if ( (lineno == 2) or (lineno >= 4) ): print( line[1:].rstrip("\n") )

def parse_cmdline():
    p = ArgumentParserWithCustomError( add_help=False )

    # Standard command line arguments

    p.add_argument( "-h", "--help",             action="store_true"                                     )
    p.add_argument( "-d", "--distribution",     choices=["uniform","reddit"],       default="uniform"   )
    p.add_argument( "-s", "--size",             choices=["all","10K","100K"],       default="all"       )
    p.add_argument( "-g", "--degree",           choices=["all","10","100","1000"],  default="all"       )
    p.add_argument( "-t", "--type",             choices=["all","int","float"],      default="all"       )    

    opts = p.parse_args()
    if opts.help: p.error()
    return opts

#-------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------

def main():
    opts = parse_cmdline()

    # select distribution
    distribution_dict = {
        'uniform'   :   ['uniform'],
        'reddit'    :   ['reddit'],
    }

    # choose size
    size_dict = {
        'all'   :   ['10K','100K'],
        '10K'   :   ['10K'],
        '100K'  :   ['100K'],
    }

    # choose average degree
    degree_dict = {
        'all'   :   ['10','100','1000'],
        '10'    :   ['10'],
        '100'   :   ['100'],
        '1000'  :   ['1000'],
    }

    # choose type
    type_dict = {
        'all'   :   ['int32','float32'],
        'int'   :   ['int32'],
        'float' :   ['float32'],
    }

    dataset_folder = '/work/shared/common/research/graphblas/data/sparse_matrix_graph/'
    home_temp = '/home/yd383/HBM-Graph/spmspv/data/'
    print('Converting graph datasets under',dataset_folder)

    for distribution in distribution_dict[opts.distribution]:
        if distribution == 'reddit' :
            dataset_name = 'reddit_200K_115M_csr_float32.npz'
            print(dataset_name)
            original_mat = scipy.sparse.load_npz(dataset_folder + dataset_name)
            print('loaded',end = ', ')
            converted_mat = original_mat.tocsc(copy=True)
            print('converted',end = ', ')
            if type_dict[opts.type] == ['int32']:
                target_name = 'reddit_200K_115M_csc_int32.npz'
                int_data = []
                for element in converted_mat.data :
                    int_data.append(int(element))
                converted_dtype_mat = scipy.sparse.csc_matrix((int_data,converted_mat.indices,converted_mat.indptr),shape=converted_mat.shape)
            else :
                target_name = 'reddit_200K_115M_csc_float32.npz'
                converted_dtype_mat = converted_mat
            scipy.sparse.save_npz(dataset_folder + target_name,converted_dtype_mat)
            print('saved to',end = ':')
            print(target_name)
        else :
            for size in size_dict[opts.size]:
                for degree in degree_dict[opts.degree]:
                    for datatype in type_dict[opts.type]:
                        dataset_name = distribution + '_' + size + '_' + degree + '_csr_' + datatype + '.npz'
                        target_name = distribution + '_' + size + '_' + degree + '_csc_' + datatype + '.npz'
                        print(dataset_name)
                        original_mat = scipy.sparse.load_npz(dataset_folder + dataset_name)
                        print('loaded',end = ', ')
                        converted_mat = original_mat.tocsc(copy=True)
                        print('converted',end = ', ')
                        scipy.sparse.save_npz(dataset_folder + target_name,converted_mat)
                        print('saved to',end = ':')
                        print(target_name)


#-------------------------------------------------------------------------
# program entry
#-------------------------------------------------------------------------
main()
