#!/usr/bin/env python

import click
from os import makedirs
import os
from os.path import join
import tempfile
import sys
import multiprocessing
from shutil import rmtree

import qiime2
from qiime2.sdk import Artifact
import q2_fragment_insertion
from q2_types.feature_data import (AlignedDNASequencesDirectoryFormat,
                                   DNASequencesDirectoryFormat)
from q2_types.tree import NewickFormat, Phylogeny, Rooted

@click.command()
@click.option('--method', type=click.Choice(['path', 'otus']), required=True)
@click.option('--cores', type=click.IntRange(1, multiprocessing.cpu_count(), clamp=True), required=False, default=1)
@click.option('--tmpdir', default=None, type=click.Path())
@click.argument('input_fragment_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
@click.option('--reference_alignment', type=click.Path(exists=True))
@click.option('--reference_phylogeny', type=click.Path(exists=True))
@click.option('--reference_name', default=None, required=False)  # this is a dummy to allow tax-credit's framework to record reference name as a parameter 
def run_sepp(input_fragment_file, output_file, method, tmpdir, cores, reference_alignment, reference_phylogeny, reference_name):
    clear_tmpdir = False

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
        clear_tmpdir = True
    if not os.path.exists(tmpdir):
        raise ValueError("Temporary working directory '%s' does not exists." % tmpdir)
    sys.stderr.write("Using '%s' as temporary working directory.\n" % tmpdir)

    # load fragments into qiime2 artifact
    ar_repseq = qiime2.Artifact.import_data(
        'FeatureData[Sequence]', input_fragment_file)

    fp_insertion_tree = join(tmpdir, 'insertion_tree.qza')
    if not os.path.exists(fp_insertion_tree):
        # load reference alignment
        if reference_alignment is not None:
            if str(reference_alignment).endswith('qza'):
                ar_ref_aln = Artifact.load(reference_alignment)
            else:
                ar_ref_aln = qiime2.Artifact.import_data(
                    "FeatureData[AlignedSequence]", reference_alignment)
        # load reference phylogeny
        if reference_phylogeny is not None:
            if str(reference_phylogeny).endswith('qza'):
                ar_ref_tree = Artifact.load(reference_phylogeny)
            else:
                ar_ref_tree = qiime2.Artifact.import_data(
                    "Phylogeny[Rooted]", reference_phylogeny)

        # run SEPP
        ar_tree, ar_placements = q2_fragment_insertion.sepp(
            ar_repseq.view(DNASequencesDirectoryFormat),
            threads=cores,
            reference_alignment=ar_ref_aln.view(AlignedDNASequencesDirectoryFormat) if reference_alignment is not None else None,
            reference_phylogeny=ar_ref_tree.view(NewickFormat) if reference_phylogeny is not None else None)
        # save tree to file
        ar_tree = Artifact.import_data(Phylogeny[Rooted], ar_tree)
        ar_tree.save(fp_insertion_tree)
    else:
        ar_tree = Artifact.load(fp_insertion_tree) #.view(NewickFormat)

    taxonomy = None
    if method == 'otus':
        # determine lineage for each rep-seq by finding closest OTU in insertion-tree
        taxonomy = q2_fragment_insertion.classify_otus_experimental(
            ar_repseq.view(DNASequencesDirectoryFormat),
            ar_tree.view(NewickFormat))
    elif method == 'path':
        # determine lineage for each rep-seq by traversing from inserted tip towards root
        # and collect taxonomic labels of reference tree
        taxonomy = q2_fragment_insertion.classify_paths(
            ar_repseq.view(DNASequencesDirectoryFormat),
            ar_tree.view(NewickFormat))

    # format results to match tax-credit
    taxonomy.index.name = '#OTU ID'
    taxonomy.rename(columns={'Taxon': 'taxonomy'}, inplace=True)
    taxonomy['confidence'] = 1.0
    taxonomy['num hits'] = 1

    # write results to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    taxonomy.to_csv(output_file, sep="\t")

    if clear_tmpdir:
        rmtree(tmpdir)


if __name__ == "__main__":
    run_sepp()
