## @knitr 1
set.seed(10)
require(quantmod)
require(fBasics)
source("yz.R")
getSymbols("AAPL",from="2007-01-03", to="2015-04-30")
AAPL["/2014-06-06",]=AAPL["/2014-06-06",]/7

open=AAPL$AAPL.Open
close=AAPL$AAPL.Close
high=AAPL$AAPL.High
low=AAPL$AAPL.Low
N=length(open)
f=(24-6.5)/24
    
S0=sqrt(252*(diff(close)^2))
S0=S0[-1,]
S1=sqrt(252*((open[2:N]-close[1:N-1])^2/(2*f)+(close[2:N]-open[2:N])^2/(2*(1-f))))
S2=sqrt(252*((high-low)^2/(4*log(2))))
S3=sqrt(252*(.17*(open[2:N]-close[1:N-1])^2/f+.83*(high[2:N]-low[2:N])^2/((1-f)*4*log(2))))
S5=sqrt(252*(.5*(high[2:N]-low[2:N])^2-(2*log(2)-1)*(close[2:N]-open[2:N])^2))
S6=sqrt(252*(.12*(open[2:N]-close[1:N-1])^2/f+.88*S5/(1-f)))

tableS=cbind(basicStats(S0),basicStats(S1),basicStats(S2),basicStats(S3),basicStats(S5),basicStats(S6))
colnames(tableS)<-c("S0","S1","S2","S3","S5","S6")
print(tableS)

## @knitr 2
open=unclass(open)
close=unclass(close)
high=unclass(high)
low=unclass(low)
m1=yz(open,high,low,close)
varyz=sqrt(252*m1$yzsq)
ts.plot(varyz,main="Time Plot of Estimated Volatility")
yy=log(varyz[64:length(varyz)])
t.test(yy)
m2=arima(yy,order=c(0,1,2))
m2
tsdiag(m2)

predictm2=predict(m2,5)
predictm2

pp=exp(predictm2$pred+0.5*predictm2$se^2)
pp

## @knitr 2o
Ioutlier=rep(0,length(m2$resid))
Ioutlier[which.max(m2$resid)]=1
m3=arima(yy,order=c(0,1,2),xreg=Ioutlier)
m3

## @knitr 3a
cokedata=read.table("m-kosp-4114.txt",header=T)
coke=log(cokedata$ko+1)
sandp=log(cokedata$sprtrn+1)
Mt=ifelse(coke>0,1,0)
St=ifelse(sandp>0,1,0)
Mtnolags=Mt[3:length(Mt)]
Mt1lag=Mt[2:(length(Mt)-1)]
Mt2lag=Mt[1:(length(Mt)-2)]
St1lag=St[2:(length(St)-1)]
St2lag=St[1:(length(St)-2)]
m1=glm(Mtnolags~Mt1lag+St1lag+Mt2lag+St2lag,family=binomial)
summary(m1)

## @knitr 3b
X=cbind(Mt1lag,St1lag,Mt2lag,St2lag)
require(nnet)
m2=nnet(X,Mtnolags,size=2,skip=T)
summary(m2)

## @knitr 3c
yf=Mtnolags[803:886]
Xf=X[803:886,]
yfit=Mtnolags[1:802]
Xfit=X[1:802,]
m4=glm(yfit~Xfit,family=binomial)

coef=m4$coefficients
Xfbind=cbind(rep(1,84),Xf)
m4p=exp(Xfbind%*%as.matrix(coef,5,1))/(1+exp(Xfbind%*%as.matrix(coef,5,1)))
logi=c(1:84)[m4p >= 0.5]
yhat=rep(0,84)
yhat[logi]=1
sum((yf-yhat)^2)

m5=nnet(Xfit,yfit,size=2,skip=T)
predictm5=predict(m5,Xf)
ypredict=ifelse(predictm5 > 0.5,1,0)
sum((yf-ypredict)^2)

## @knitr 4
db=read.table("taq-goog-may1t152013.txt",header=T)
source("hfrtn.R")
m6=hfrtn(db,5)

ts.plot(m6$rtn,main="5-m returns of XOM")
Box.test(m6$rtn,lag=10,type='Ljung')

RV=NULL
for (i in 1:11){
  daycount=(i-1)*77
  x=sum(m6$rtn[(daycount+1):(daycount+77)]^2)
  RV=c(RV,x)
}

m7=hfrtn(db,1)
RV1=NULL
for (i in 1:11){
  daycount=(i-1)*389
  x=sum(m7$rtn[(daycount+1):(daycount+389)]^2)
  RV1=c(RV1,x)
}

tableRV=cbind(RV,RV1)
colnames(tableRV)<-c("RV","RV1")

par(mfcol=c(2,1))
ts.plot(RV,main="RV: 5-m log returns")
ts.plot(RV1,main="RV: 1-m log returns")

## @knitr 5
source("hfntra.R")
m8=hfntra(db,5)
par(mfcol=c(1,1))
ts.plot(m8$ntrad,main="Numbers of trade in 5-m")
acf(m8$ntrad,lag.max=310)
