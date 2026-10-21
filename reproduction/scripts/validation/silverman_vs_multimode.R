# Score the same samples with the reference implementation.
#   install.packages("multimode")
#   Rscript silverman_vs_multimode.R
# method="SI" is Silverman (1981); submethod=1 is the RESCALED bootstrap, which
# is what our Python implements. mod0=k is "at most k modes" under H0.
suppressMessages(library(multimode))
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])))
x <- read.csv(file.path(here, "validation_samples.csv"))
B <- 999
out <- data.frame()
for (nm in names(x)) {
  v <- x[[nm]]
  for (k in 1:3) {
    set.seed(4242 + k)
    p <- modetest(v, mod0 = k, method = "SI", submethod = 1, B = B)$p.value
    cat(sprintf("  %-18s k=%d  p=%.4f\n", nm, k, p))
    out <- rbind(out, data.frame(sample = nm, k = k, p_multimode = p))
  }
}
write.csv(out, file.path(here, "validation_multimode.csv"), row.names = FALSE)
cat("\nwrote validation_multimode.csv -- now run: python3 silverman_vs_multimode.py --compare\n")
