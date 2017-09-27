base_dir <- commandArgs(TRUE)[1]

if (is.na(base_dir)) {
  stop("Invalid number of arguments.\n")
}

day.list <- list(
  training2nd = seq.Date(from = as.Date("2013-06-06"), by = 1, length.out = 7),
  training3rd = seq.Date(from = as.Date("2013-10-19"), by = 1, length.out = 9)
)

for(season in c("training2nd", "training3rd")) {
  for (i in seq_along(day.list[[season]])) {
    day <- day.list[[season]][i]
    yyyymmdd <- format(day, "%Y%m%d")
    input_path <- sprintf("%s/cache/bidimpclk.%s.sim2.Rds", base_dir, yyyymmdd)
    output_path <- sprintf("./data/bidimpclk.%s.sim2.csv", yyyymmdd)
    cat("Reading", input_path, "...\n")
    bidimpclk <- readRDS(input_path)
    cat("Writing to", output_path, "...\n")
    write.csv(bidimpclk, output_path)
  }
}
