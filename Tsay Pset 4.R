## @knitr 1.a
da=read.table("d-amzn3dx0914.txt", header=T)
library(fGarch)
source("Igarch.R")
source("garchM.R")
source("Tgarch11.R")
source("Tsats.R")
rt=log(da$amzn+1)*100 
t.test(rt)
par(mfcol=c(2,1))
acf(rt)
pacf(rt)
Box.test(rt,lag=10,type="Ljung")

## @knitr 1.b
arima(rt,order=c(0,0,1))$aic
arima(rt,order=c(1,0,0))$aic
arima(rt,order=c(1,0,1))$aic

m1=garchFit(~arma(1,1)+garch(1,1),data=rt,trace=F)
m1@fit$ics
par(mfcol=c(1,1))
plot(m1,which=13)

## @knitr 1.c
m2=garchFit(~arma(1,1)+garch(1,1),data=rt,trace=F,cond.dist="std")
m2@fit$ics
plot(m2,which=13)

## @knitr 1.d
pm3=predict(m2,5)
pm3

## @knitr 2.a
at=rt-mean(rt)
m3=Igarch(at,volcnt=T)

## @knitr 2.b
names(m3)
sresid=at/m3$volatility
Box.test(sresid,lag=10,type="Ljung")

## @knitr 2.c
Box.test(sresid^2,lag=10,type="Ljung")

## @knitr 2.d
v1=(1-0.990)*at[length(at)]^2+.990*m3$volatility[length(at)]^2
sqrt(v1)

## @knitr 3.a
db=read.table("m-mcd3dx6614.txt", header=T)
logret=log(db$mcd+1)
t.test(logret)
Box.test(logret,lag=12,type="Ljung")
Box.test((logret-mean(logret))^2,lag=12,type="Ljung")

## @knitr 3.b
m4=garchFit(~garch(1,1),data=logret,trace=F)
m4@fit$ics
plot(m4,which=13)

## @knitr 3.c
m5=Igarch(logret)

## @knitr 3.d
m6=garchFit(~garch(1,1),data=logret,trace=F,cond.dist="sstd")
m6@fit$ics
plot(m6,which=13)

## @knitr 3.e
(.833-1)/.0507

## @knitr 3.f
m7=garchM(logret)

## @knitr 3.g
m8=Tgarch11(logret)

## @knitr 4.a
logsimp=log(db$vwretd+1)
t.test(logsimp)
Box.test(logsimp,lag=12,type="Ljung")
Box.test((logsimp-mean(logsimp))^2,lag=12,type="Ljung") #ARCH Effects
m9=garchFit(~garch(1,1),data=logsimp,trace=F)
m9@fit$ics
plot(m9,which=13)
m10=garchFit(~garch(1,1),data=logsimp,trace=F,cond.dist="std")
m10@fit$ics
plot(m10,which=13)
m11=garchFit(~garch(1,1),data=logsimp,trace=F,cond.dist="sstd")
m11@fit$ics
plot(m11,which=13)
(.7432-1)/.0486

## @knitr 4.b
predict(m11,5)

## @knitr 4.c
m12=garchFit(~aparch(1,1),data=logsimp, trace=F,delta=2,include.delta=F, cond.dist="sstd")
m12@fit$ics
plot(m12,which=13)

## @knitr 5.a
dc=read.table("d-exusuk-0615.txt", header=T)
exch=log(dc$Value)
acf(exch) #Unit Root
dex=diff(exch)
par(mfcol=c(2,1))
acf(dex)
pacf(dex)
t.test(dex)
Box.test(dex,lag=20,type="Ljung")

ar(dex,method="mle")$order
m13=arima(dex,order=c(10,0,0),include.mean=F)
m13
tsdiag(m13,gof=20)
arima(dex,order=c(1,0,1),include.mean=F)
c1=Tstats(m13)
m14=arima(dex,order=c(10,0,0),include.mean=F,fixed=c1)
m14

m15=garchFit(~arma(10,1)+garch(1,1),data=dex,trace=F)
m15@fit$ics
par(mfcol=c(1,1))
plot(m15,which=13)
m16=garchFit(~garch(1,1),data=dex, trace=F, cond.dist="std", include.mean=F)
m16@fit$ics
plot(m16,which=13)

## @knitr 5.b
dex100=dex*100
m17=garchFit(~aparch(1,1),data=dex100, trace=F,delta=2,include.delta=F, cond.dist="sstd",include.mean=F)
m17@fit$ics
(.908-1)/.0266
plot(m17,which=13)
m19=garchFit(~garch(1,1),data=dex100,trace=F,cond.dist="std",leverage=T,include.mean=F)
m19@fit$ics
plot(m19,which=13)
