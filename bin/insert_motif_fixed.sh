#! /bin/sh
# Insert a sequence randomly in each sequence in FASTA files
if [ "$#" -lt 1 ];then
	echo "Usage: $0 motif [fasta_file]"
	exit 1
fi

awk -v motif=$1 'BEGIN{srand()}
/^>/{print;next} 
{
	pos = int((length($0) - length(motif))*rand()) + 1;
	print substr($0, 1, pos - 1) motif substr($0, pos + length(motif))
}' $2
