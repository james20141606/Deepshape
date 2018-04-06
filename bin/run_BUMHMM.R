args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 3){
    stop(sprintf("Usage: run_BUMHMM.R filename outfile min_coverage"))
}
filename <- args[1]
outfile <- args[2]
minCoverage <- as.integer(args[3])

nuclNum <- 3

suppressPackageStartupMessages({
library(BUMHMM)
library(Biostrings)
library(SummarizedExperiment)
library(rhdf5)
})

run_BUMHMM <- function(filename, outfile){
    message('Read data from file: ', filename)
    sequence <- DNAString(h5read(filename, '/sequence'))
    myassays <- list()
    myassays[['coverage']] <- h5read(filename, '/coverage')
    myassays[['dropoff_count']] <- h5read(filename, '/dropoff_count')
    myassays[['dropoff_rate']] <- myassays[['dropoff_count']]/myassays[['coverage']]
    myassays[['dropoff_rate']][is.na(myassays[['dropoff_rate']])] <- 0
    rowdata <- DataFrame(nucl=Views(sequence, successiveIRanges(rep(1, nchar(sequence)))))
    coldata <- DataFrame(replicate=h5read(filename, '/replicate'),
                         row.names=h5read(filename, '/sample_name'))
    se <- SummarizedExperiment(assays=myassays, colData=coldata, rowData=rowdata)
    Nc <- sum(coldata[, 1] == 'control')
    Nt <- sum(coldata[, 1] == 'treatment')
    nuclSelection <- selectNuclPos(se, Nc, Nt, minCoverage)
    assay(se, 'dropoff_rate') <- scaleDOR(se, nuclSelection, Nc, Nt)
    message('Compute streches')
    stretches <- computeStretches(se, minCoverage)
    message('Correct bias in variance')
    varStab <- stabiliseVariance(se, nuclSelection, Nc, Nt)
    LDR_C <- varStab$LDR_C
    LDR_CT <- varStab$LDR_CT
    if(nchar(sequence) > 10000){
        message('Find nucleotide patterns')
        patterns <- nuclPerm(nuclNum)
        nuclPosition <- findPatternPos(patterns, sequence, '+')
    } else{
        nuclPosition <- list()
        nuclPosition[[1]] <- 1:nchar(sequence)
    }
    posteriors <- computeProbs(LDR_C, LDR_CT, Nc, Nt, '+', nuclPosition,
                               nuclSelection$analysedC, nuclSelection$analysedCT,
                               stretches)
    shifted_posteriors <- matrix(, nrow=dim(posteriors)[1], ncol=1)
    shifted_posteriors[1:(length(shifted_posteriors) - 1)] <- posteriors[2:dim(posteriors)[1], 2]

    message('write posteriors to file ', outfile)
    h5createFile(outfile)
    h5write(shifted_posteriors[,1], outfile, 'posteriors')
}

tryCatch({
    run_BUMHMM(filename, outfile)
}, error=function(e){warning('Ignoring error: ', conditionMessage(e))})
