correlation <- function (M, N) {
   data <- matrix(0, nrow=N, ncol=M);
   for (i in 1:N)
      for (j in 1:M)
	     data[i,j] = ((i-1)*(j-1))/M+i;

    cor <- cor(data);
}

covariance <- function (M, N) {
   data <- matrix(0, nrow=N, ncol=M);
   for (i in 1:N)
      for (j in 1:M)
	     data[i,j] = ((i-1)*(j-1))/M;
		 
    res <- cov(data);
}

gemm <- function(NI, NJ, NK) {
   alpha <- 1.5;
   beta  <- 1.2;
   C <- matrix(0, nrow=NI, ncol=NJ);
   A <- matrix(0, nrow=NI, ncol=NK);
   B <- matrix(0, nrow=NK, ncol=NJ);
   
   for (i in 1:NI)
      for (j in 1:NJ)
	     C[i,j] <- (((i-1)*(j-1)+1) %% NI) / NI;
		 
   for (i in 1:NI)
      for (j in 1:NK)
	     A[i,j] <- (((i-1)*j) %% NK) / NK;
   
   for (i in 1:NK)
      for (j in 1:NJ)
	     B[i,j] <- (((i-1)*(j+1)) %% NJ) / NJ;
		 
  res <- alpha *A %*% B + beta * C;
   
}

gemver <- function (N) {
   alpha <- 1.5;
   beta <- 1.2;
   A <- matrix(0, nrow=N, ncol=N);
   u1 <- numeric(N);
   u2 <- numeric(N);
   v1 <- numeric(N);
   v2 <- numeric(N);
   y  <- numeric(N);
   z  <- numeric(N);
   
   for (i in 1:N) {
      ip <- i-1;
	  u1[i] <- ip;
	  u2[i] <- ((ip+1)/N)/2.0;
	  v1[i] <- ((ip+1)/N)/4.0;
	  v2[i] <- ((ip+1)/N)/6.0;
	  y[i]  <- ((ip+1)/N)/8.0;
	  z[i]  <- ((ip+1)/N)/9.0;
	  
      for (j in 1:N) {
	      A[i,j] <- (((i-1)*(j-1)) %% N) / N;
	  }
   }
 

   Aprime <- A + u1 %o% v1 + u2 %o% v2;	
   x <- beta * t(Aprime) %*% y + z;	
   w <- alpha * Aprime %*% x;	
   
   list(Aprime, x, w);
}

gesummv <- function (N) {
   alpha <- 1.5;
   beta <- 1.2;
   A <- matrix(0, nrow=N, ncol=N);
   B <- matrix(0, nrow=N, ncol=N);
   x  <- numeric(N);
   
   for (i in 1:N) {
      x[i] <- ((i-1) %% N) / N;
	  
      for (j in 1:N) {
	      A[i,j] <- (((i-1)*(j-1)+1) %% N) / N;
	      B[i,j] <- (((i-1)*(j-1)+2) %% N) / N;
	  }
   }
 
   y <- alpha * A %*% x + beta * B %*% x;
}


symm <- function(M, N) {
   alpha <- 1.5;
   beta <- 1.2;
   A <- matrix(0, nrow=M, ncol=M);
   B <- matrix(0, nrow=M, ncol=N);
   C <- matrix(0, nrow=M, ncol=N);
   
   for (i in 1:M)
      for (j in 1:N) {
         C[i,j] <- ((i+j-2) %% 100) / M;
         B[i,j] <- ((N+i-j) %% 100) / M;
	  }
   for (i in 1:M)
     for (j in i:M) {
         A[i,j] <- ((i+j-2) %% 100) / M;
		 A[j,i] <- A[i,j];
     }
		 
    res <- alpha * A %*% B + beta * C;
   
}

syrk <- function(M, N) {

   alpha <- 1.5;
   beta <- 1.2;
   A <- matrix(0, nrow=N, ncol=M);
   C <- matrix(0, nrow=N, ncol=N);
   
   for (i in 1:N)
      for (j in 1:M)
	     A[i,j] <- (((i-1)*(j-1)+1) %% N) / N;
		 
   for (i in 1:N)
      for (j in 1:N)
	     C[i,j] <- (((i-1)*(j-1)+2) %% M) / M;
		 
	res <- alpha * A %*% t(A) + beta * C;

#note that syrk only stores lower triangular part of the resulting symmetric matrix
}

syr2k <- function(M, N) {

   alpha <- 1.5;
   beta <- 1.2;
   A <- matrix(0, nrow=N, ncol=M);
   B <- matrix(0, nrow=N, ncol=M);
   C <- matrix(0, nrow=N, ncol=N);
   
   for (i in 1:N)
      for (j in 1:M) {
	     A[i,j] <- (((i-1)*(j-1)+1) %% N) / N;
	     B[i,j] <- (((i-1)*(j-1)+2) %% M) / M;
	  }
		 
   for (i in 1:N)
      for (j in 1:N)
	     C[i,j] <- (((i-1)*(j-1)+3) %% N) / M;
		 
	res <- alpha * A %*% t(B) + alpha * B %*% t(A) + beta * C;

}


trmm <- function(M, N) {

   alpha <- 1.5;
   A <- matrix(0, nrow=M, ncol=M);
   B <- matrix(0, nrow=M, ncol=N);
   
   for (i in 1:M) {
      for (j in 1:i-1) {
	     A[i,j] <- (((i-1)+(j-1)) %% M) / M;
	  }
	  A[i,i] <- 1.0;
      for (j in 1:N)
	     B[i,j] <- ((N+(i-1)-(j-1)) %% N) / N;
   }
	
		 
	res <- alpha * t(A) %*% B;
}

twomm <- function(NI, NJ, NK, NL) {
   alpha <- 1.5;
   beta <- 1.2;
   A <- matrix(0, nrow=NI, ncol=NK);
   B <- matrix(0, nrow=NK, ncol=NJ);
   C <- matrix(0, nrow=NJ, ncol=NL);
   D <- matrix(0, nrow=NI, ncol=NL);
   
   for (i in 1:NI)
      for (j in 1:NK)
	     A[i,j] <- (((i-1)*(j-1)+1) %% NI) / NI;
		 
   for (i in 1:NK)
      for (j in 1:NJ)
	     B[i,j] <- (((i-1)*(j)) %% NJ) / NJ;
		 
   for (i in 1:NJ)
      for (j in 1:NL)
	     C[i,j] <- (((i-1)*(j+2)+1) %% NL) / NL;
		 
   for (i in 1:NI)
      for (j in 1:NL)
	     D[i,j] <- (((i-1)*(j+1)) %% NK) / NK;
		 
		 
    res <- alpha * A %*% B %*% C + beta * D;		 

}


threemm <- function(NI, NJ, NK, NL, NM) {
   A <- matrix(0, nrow=NI, ncol=NK);
   B <- matrix(0, nrow=NK, ncol=NJ);
   C <- matrix(0, nrow=NJ, ncol=NM);
   D <- matrix(0, nrow=NM, ncol=NL);
   
   for (i in 1:NI)
      for (j in 1:NK)
	     A[i,j] <- (((i-1)*(j-1)+1) %% NI) / (5*NI);
		 
   for (i in 1:NK)
      for (j in 1:NJ)
	     B[i,j] <- (((i-1)*(j)+2) %% NJ) / (5*NJ);
		 
   for (i in 1:NJ)
      for (j in 1:NM)
	     C[i,j] <- (((i-1)*(j+2)) %% NL) / (5*NL);
		 
   for (i in 1:NM)
      for (j in 1:NL)
	     D[i,j] <- (((i-1)*(j+1)+2) %% NK) / (5*NK);
		 
		 
    E <- A %*% B;
    F <- C %*% D;
	G <- E %*% F;

}

atax <- function(M, N) {
   A <- matrix(0, nrow=M, ncol=N);
   x <- numeric(N);
   
   for (i in 1:N)
     x[i] <- 1 + ((i-1) / N);

   for (i in 1:M)
      for (j in 1:N) 
         A[i,j] <- (((i-1)+(j-1)) %% N) / (5*M)
	 
   
   y <- t(A) %*% (A %*% x);
   
}

mvt <- function(N) {
   x1 <- numeric(N);
   x2 <- numeric(N);
   y1 <- numeric(N);
   y2 <- numeric(N);
   A <- matrix(0, nrow=N, ncol=N);
   
   for (i in 1:N) {
      ip <- i-1;
      x1[i] <- (ip %% N) / N;
      x2[i] <- ((ip + 1) %% N) / N;
      y1[i] <- ((ip + 3) %% N) / N;
      y2[i] <- ((ip + 4) %% N) / N;
	  for (j in 1:N)
	     A[i,j] <- (((i-1)*(j-1)) %% N) / N;
   }
   
   
   x1 <- x1 + A %*% y1;
   x2 <- x2 + t(A) %*% y2;
}

cholesky <- function(N) {
   A <- matrix(0, nrow=N, ncol=N)
   
   for (i in 1:N) {
      for (j in 1:i)
	     A[i,j] <- ((-((j-1) %% N)) / N) + 1
	   A[i,i] <- 1;
	}
	
	A <- A %*% t(A);
	
	res <- chol(A);
}

durbin <- function(N) {
   r <- numeric(N+1)

   r[1] = 1;
   for (i in 2:(N+1)) 
      r[i] <- N+1-(i-2)
	  
   r1 <- r[1:N]
   r2 <- r[2:(N+1)]
   
   res <- solve(toeplitz(r1), -r2)
}

gramschmidt <- function(M, N) {
   A <- matrix(0, nrow=M, ncol=N);

   for (i in 1:M)
      for (j in 1:N)
	    A[i,j] <- ((((i-1)*(j-1)) %% M) / M)*100 + 10
		
    res <- qr(A);
	
	list(res, qr.Q(res), qr.R(res))
}

library('Matrix')
lu.polybench <- function(N) {
   A <- matrix(0, nrow=N, ncol=N)

   
   for (i in 1:N) {
      for (j in 1:i)
	     A[i,j] <- ((-((j-1) %% N)) / N) + 1
	   A[i,i] <- 1;
	}
	
	A <- A %*% t(A);
	
	res <- expand(lu(A));
}

ludcmp <- function(N) {
   b <- numeric(N);
   for (i in 1:N)
      b[i] <- ((i/N)/2.0) +4;
   
   A <- matrix(0, nrow=N, ncol=N)
   
   for (i in 1:N) {
      for (j in 1:i)
	     A[i,j] <- ((-((j-1) %% N)) / N) + 1
	   A[i,i] <- 1;
	}
	
	A <- A %*% t(A);
	
	res <- solve(A, b);
}

trisolv <- function(N) {
   L <- matrix(0, nrow=N, ncol=N)
   b <- c(0:(N-1))
   
   for (i in 1:N)
      for (j in 0:i)
	     L[i,j] <- (((i-1)+N-(j-1)+1)*2) / N
		 
	res <- solve(L, b)
}
