# Variable bounds

The initial population is sampled randomly in the interval of each variable, so finite variable bounds are strongly recommended. If a variable has an infinite bound, the wrapper emits a warning and samples that variable in a finite surrogate interval instead of failing.
