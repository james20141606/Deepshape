#! /usr/bin/env python
from cmdtool import CommandLineTool, Argument
from ioutils import prepare_output_file
from formats import read_fasta
import sys, os
import logging

def icshape_raw_rt_to_genomic_data(infile, logger=None):
    import numpy as np
    from genomic_data import GenomicData
    if not logger:
        logger = logging.getLogger('icshape_raw_rt_to_genomic_data')

    logger.info('read input rt file: ' + infile)
    name = []
    length = []
    rpkm = []
    rt_stop = []
    base_density = []
    with open(infile, 'r') as f:
        f.readline()
        n_records = 0
        for lineno, line in enumerate(f):
            c = line.strip().split('\t')
            if (lineno % 2) == 0:
                name.append(c[0])
                length.append(int(c[1]))
                rpkm.append(float(c[2].split(',')[0]))
                base_density.append(np.asarray(c[3:], dtype='float').astype('int32'))
            else:
                rt_stop.append(np.asarray(c[3:], dtype='float').astype('int32'))
            n_records += 1
    logger.info('successfully read %d records'%n_records)

    data = GenomicData.from_data(name,
        features={'rt_stop': rt_stop,
                  'base_density': base_density},
        meta={'rpkm': np.asarray(rpkm, dtype='float64'),
              'length': np.asarray(length, dtype='int64')}
        )
    return data

class IcshapeRawRtToGenomicData(CommandLineTool):
    description = 'Convert libX.rt files to GenomicData format'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True,
            help='input icSHAPE rt file'),
        Argument('outfile', short_opt='-o', type=str, required=True,
            help='output GenomicData file')]
    def __call__(self):
        from genomic_data import GenomicData
        import numpy as np

        self.logger.info('read input rt file: ' + self.infile)
        name = []
        length = []
        rpkm = []
        rt_stop = []
        base_density = []
        with open(self.infile, 'r') as f:
            f.readline()
            n_records = 0
            for lineno, line in enumerate(f):
                c = line.strip().split('\t')
                if (lineno % 2) == 0:
                    name.append(c[0])
                    length.append(int(c[1]))
                    rpkm.append(float(c[2].split(',')[0]))
                    rt_stop.append(np.asarray(c[3:], dtype='float').astype('int32'))
                else:
                    base_density.append(np.asarray(c[3:], dtype='float').astype('int32'))
                n_records += 1
        self.logger.info('successfully read %d records'%n_records)

        self.logger.info('create output file: ' + self.outfile)
        prepare_output_file(self.outfile)
        GenomicData.from_data(name,
            features={'rt_stop': rt_stop,
                      'base_density': base_density},
            meta={'rpkm': np.asarray(rpkm, dtype='float64'),
                  'length': np.asarray(length, dtype='int64')}
            ).save(self.outfile)

def run_BUMHMM(rt_stop_control, coverage_control,
        rt_stop_treatment, coverage_treatment, seq):
    pass

def run_BUMHMM_for_icSHAPE(control_files, treatment_files, sequence_file):
    control_data = []
    for filename in control_files:
        control_data.append(GenomicData.load(filename))
    treatment_data = []
    for filename in treatment_files:
        treatment_data.append(GenomicData.load(filename))
    sequences = dict(read_fasta(sequence_file))
    names, counts = np.unique(np.concatenate(map(lambda x: x.names, control_data) + map(lambda x: x.names, treatment_data)), return_counts=True)
    common_names = names[counts >= (len(control_data) + len(treatment_data))]

    for name in common_names:
        run_BUMHMM(rt_stop_control=map(lambda x: x.feature(name, 'rt_stop'), control_data),
                   coverage_control=map(lambda x: x.feature(name, 'base_density'), control_data),
                   rt_stop_treatment=map(lambda x: x.feature(name, 'rt_stop'), treatment_data),
                   coverage_treatment=map(lambda x: x.feature(name, 'base_density'), treatment_data),
                   seq=sequences[name])

class PrepareBumhmmForIcshape(CommandLineTool):
    description = 'Prepare data for running BUMHMM'
    arguments = [Argument('control_file', type=str, action='append', required=True),
        Argument('treatment_file', type=str, action='append', required=True),
        Argument('sequence_file', type=str, required=True),
        Argument('outfile', short_opt='-o', type=str, required=True)]
    def __call__(self):
        import numpy as np
        import h5py

        control_data = []
        for filename in self.control_file:
            control_data.append(icshape_raw_rt_to_genomic_data(filename, self.logger))
        treatment_data = []
        for filename in self.treatment_file:
            treatment_data.append(icshape_raw_rt_to_genomic_data(filename, self.logger))
        combined_data = control_data + treatment_data
        self.logger.info('read sequence file: ' + self.sequence_file)
        sequences = dict(read_fasta(self.sequence_file))
        names, counts = np.unique(np.concatenate(map(lambda x: x.names, combined_data)), return_counts=True)
        common_names = names[counts >= len(combined_data)]

        self.logger.info('create output file: ' + self.outfile)
        prepare_output_file(self.outfile)
        fout = h5py.File(self.outfile, 'w')
        ncol = len(control_data) + len(treatment_data)
        sample_name = np.asarray(['C%d'%i for i in range(len(control_data))] + ['T%d'%i for i in range(len(treatment_data))], dtype='S')
        replicate = np.asarray(['control']*len(control_data) + ['treatment']*len(treatment_data), dtype='S')
        """
        for i, name in enumerate(common_names):
            self.logger.info('create group: ' + str(name))
            g = fout.create_group(name)
            coverage = np.vstack(map(lambda x: x.feature('base_density', name)[1:], combined_data))
            dropoff_count = np.vstack(map(lambda x: x.feature('rt_stop', name)[:-1], combined_data))
            rpkm = np.mean(map(lambda x: x.feature('rpkm', name), combined_data))
            g.create_dataset('coverage', data=coverage)
            g.create_dataset('dropoff_count', data=dropoff_count)
            g.create_dataset('sequence', data=np.asarray(sequences[name], dtype='S'))
            g.create_dataset('sample_name', data=sample_name)
            g.create_dataset('replicate', data=replicates)
            g.create_dataset('rpkm', data=rpkm)
        """
        coverage = [[]]*len(combined_data)
        dropoff_count = [[]]*len(combined_data)
        for i in range(len(combined_data)):
            coverage[i] = [None]*len(common_names)
            dropoff_count[i] = [None]*len(common_names)
            for j in range(len(common_names)):
                coverage[i][j] = combined_data[i].feature('base_density', common_names[j])[1:]
                coverage[i][j][:20] = 0
                dropoff_count[i][j] = combined_data[i].feature('rt_stop', common_names[j])[:-1]
                dropoff_count[i][j][:20] = 0
            if i == 0:
                length = np.asarray(map(len, coverage[i]), dtype='int64')
                end = np.cumsum(length)
                start = end - length
            coverage[i] = np.concatenate(coverage[i])
            dropoff_count[i] = np.concatenate(dropoff_count[i])
        coverage = np.vstack(coverage)
        dropoff_count = np.vstack(dropoff_count)
        sequence = np.asarray(''.join(map(lambda name: sequences[name], common_names)), dtype='S')

        fout.create_dataset('name', data=common_names)
        fout.create_dataset('start', data=start)
        fout.create_dataset('end', data=end)
        fout.create_dataset('coverage', data=coverage)
        fout.create_dataset('dropoff_count', data=dropoff_count)
        fout.create_dataset('sequence', data=sequence)
        fout.create_dataset('replicate', data=replicate)
        fout.create_dataset('sample_name', data=sample_name)
        fout.close()

class BumhmmToGenomicData(CommandLineTool):
    description = 'Combine BUMHMM output files into one GenomicData file'
    arguments = [Argument('posterior_file', short_opt='-i', type=str, required=True,
            help='BUMHMM output file'),
        Argument('bumhmm_input_file', type=str, required=True),
        Argument('outfile', short_opt='-o', type=str, required=True,
            help='output file')]
    def __call__(self):
        from genomic_data import GenomicData
        import numpy as np
        import h5py

        self.logger.info('read BUMHMM file: ' + self.posterior_file)
        posteriors = h5py.File(self.posterior_file, 'r')['posteriors'][:]
        self.logger.info('read BUMHMM input file: ' + self.bumhmm_input_file)
        f = h5py.File(self.bumhmm_input_file, 'r')
        start = f['start'][:]
        end = f['end'][:]
        name = f['name'][:]
        f.close()
        values = map(lambda i: posteriors[start[i]:end[i]], range(len(name)))
        self.logger.info('save file: ' + self.outfile)
        prepare_output_file(self.outfile)
        GenomicData.from_data(name, features={'bumhmm': values}).save(self.outfile)

class IcshapeRtToWav(CommandLineTool):
    arguments = [Argument('rt_file', short_opt='-i', type=str, required=True),
        Argument('outdir', short_opt='-o', type=str, required=True)]
    def __call__(self):
        import wave
        import numpy as np
        rt = icshape_raw_rt_to_genomic_data(self.rt_file, self.logger)

        def modulate(values, wav_file, sample_rate=44100, n_channels=2, max_amp=32767, x_freq=20):
            upsample_rate = float(sample_rate)/x_freq
            T = float(len(values))/x_freq
            n_samples = int(sample_rate*T)
            x = np.empty(n_samples, dtype='float32')
            for i in range(len(values)):
                x[int(upsample_rate*i):int(upsample_rate*(i + 1))] = np.log(values[i] + 1)
            t = np.linspace(0, T, n_samples)
            y = max_amp*np.sin(2*880*np.pi*t)
            y *= x
            y *= float(max_amp)/np.abs(y.max())
            data = np.empty(n_samples*n_channels, dtype='int16')
            channel_index = np.arange(0, n_samples*n_channels, n_channels)
            data[channel_index] = y
            data[channel_index + 1] = data[channel_index]

            wav = wave.open(wav_file, 'wb')
            wav.setnchannels(n_channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.setnframes(n_samples)
            wav.setcomptype('NONE', 'no compression')
            wav.writeframes(np.getbuffer(data))
            wav.close()

        for i in np.argsort(-rt.meta['rpkm'])[:10]:
            name = rt.names[i]
            values = rt.feature('rt_stop', name)

            wav_file = os.path.join(self.outdir, '%s.wav'%name)
            self.logger.info('create wav file: ' + wav_file)
            prepare_output_file(wav_file)
            modulate(values, wav_file)


if __name__ == '__main__':
    CommandLineTool.from_argv()()
