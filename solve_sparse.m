function x = solve_sparse()
    load('A.mat', 'A');
    load('b.mat', 'b');
    load('N.mat', 'N');
    
    A = double(A);
    b = double(b);
    N = double(N);
    x = (N*A)\(N*b);
end
