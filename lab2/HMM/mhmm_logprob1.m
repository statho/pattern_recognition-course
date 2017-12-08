function [loglik, errors] = mhmm_logprob(data, prior, transmat, mu, Sigma, mixmat)
% LOG_LIK_MHMM Compute the log-likelihood of a dataset using a (mixture of) Gaussians HMM
% [loglik, errors] = log_lik_mhmm(data, prior, transmat, mu, sigma, mixmat)
%
% data{m}(:,t) or data(:,t,m) if all cases have same length
% errors  is a list of the cases which received a loglik of -infinity
%
% Set mixmat to ones(Q,1) or omit it if there is only 1 mixture component


    % initial guess of parameters
    prior0 = [1 ,zeros(1,Ns-1)];
    transmat0=zeros(Ns,Ns);
    for i=1:Ns
        for j=1:Ns
            if (j==i+1||j==i)
                transmat0(i,j)=rand;
            end
        end
    end


    [mu0, Sigma0] = mixgauss_init(Ns*M, data, cov_type);
    mu0 = reshape(mu0, [O Ns M]);
    Sigma0 = reshape(Sigma0, [O O Ns M]);
    mixmat0 = mk_stochastic(rand(Ns,M));


    [LL(dig,:), prior1(dig,:), transmat1(dig,:,:), mu1(dig,:,:,:), Sigma1(dig,:,:,:,:), mixmat1(dig,:,:)] = ...
        mhmm_em(data, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 5);

    mixmatt=reshape(mixmat1(dig,:,:),[Ns,M]);
    priorr = reshape(prior1(dig,:), [1,size(prior1,2)]);
    transmatt = reshape(transmat1(dig,:,:), [size(transmat1,2), size(transmat1,3)]);
    muu = reshape(mu1(dig,:,:,:), [size(mu1,2),size(mu1,3),size(mu1,4)]);
    Sigmaa = reshape(Sigma1(dig,:,:,:,:), [size(Sigma1,2), size(Sigma1,3), size(Sigma1,4), size(Sigma1,5)]);
    
    loglik(dig) = mhmm_logprob(data, priorr, transmatt, muu, Sigmaa, mixmatt);

end
Q = length(prior);
if size(mixmat,1) ~= Q % trap old syntax
  error('mixmat should be QxM')
end
if nargin < 6, mixmat = ones(Q,1); end

if ~iscell(data)
  data = num2cell(data, [1 2]); % each elt of the 3rd dim gets its own cell
end
ncases = length(data);

loglik = 0;
errors = [];
for m=1:ncases
  obslik = mixgauss_prob(data{m}, mu, Sigma, mixmat);
  [alpha, beta, gamma, ll] = fwdback(prior, transmat, obslik, 'fwd_only', 1);
  if ll==-inf
    errors = [errors m];
  end
  loglik = loglik + ll;
end
end