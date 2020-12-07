import sys, os

folder = sys.argv[1]

files = [x.strip() for x in open('%s_data_labeled.txt'%folder)]
metadir = 'MOSEI/Transcript/Segmented/Combined'
out_file = open('%s_data_spread.txt'%folder, 'w')

for entry in files:
    fn = entry.split()[0]
    meta = open(os.path.join(metadir, "%s.txt"%fn)).readlines()
    length = meta[-1].split('___')[3]
    length = float(length)
    conv = lambda x: round(float(x)/length,4)
    for line in meta:
        line = line.split('___')
        start, end = map(conv,line[2:4])
        out_str = "%s %f %f\n"%(entry, start, end)
        out_file.write(out_str)
out_file.close()
