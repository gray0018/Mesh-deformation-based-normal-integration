function x = solve_sparse(A_path, b_path)
    load(A_path, 'A');
    load(b_path, 'b');
    A = double(A);
    b = double(b);
    x = A\b;
end
