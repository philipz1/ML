## @knitr 1.c
library(stats4)
y <- c(0.4, 0.378, 0.356, 0.333, 0.311, 0.311, 0.289, 0.267, 0.244, 0.244, 0.222, 0.222, 0.222, 0.222, 0.222, 0.2, 0.178, 0.156)
y=y*sqrt(45)*asin(2*y-1)
s2 <- mean(y)*(1 - mean(y))/45
shrink= 1 - 15*s2/(17*var(y))
shrink

LL <- function(mu, sigma) {
  R = suppressWarnings(dnorm(y, mu, sigma))
  -sum(log(R))
}
mle(minuslogl = LL, start = list(mu = 1, sigma = 1))

sig=0.09521806 
sum((y-sig)^2) # MLE

JS0=(1-(18-2)*s2/(sum(y^2)))*y
sum((y-JS0)^2) # JS0

JSv=mean(y)+(1-(18-2)*s2/(sum((y-mean(y))^2)))*(y-mean(y))
sum((y-JSv)^2) # vbar

## @knitr 1.d

ms5=read.table("meanshift5.txt",header=F)
ms10=read.table("meanshift10.txt",header=F)
kernel = function(kern) { function(x, mean, h) {(1/h) * kern((x-mean)/h)}}
kernel1 = function(kern) { function(x, mean, h) {(1/h) * kern((w0-X)/h)}}
X=(unlist(ms5))
X=seq(1,100,1)
x = seq(-10,10,length.out=500)
h = .05; fhat.gauss = sapply(x, function(x) { mean(sapply(X, function(X) {kernel(dnorm)(x, mean=X, h)}))})
plot(x, fhat.gauss, 'l', col='blue', lwd=2)
w0=X[1]
h = .05; fhat.gauss = sapply(x, function(x) { mean(sapply(X, function(X) {kernel1(dnorm)(x, mean=X, h)}))})
w1=fhat.gauss
w1
for (ii in X){
  w0=ii
  w1=
}
