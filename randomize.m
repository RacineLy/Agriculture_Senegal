function w = randomize(Lin, Lout, epsilon)
  
  w = rand(Lout, Lin+1)*(2*epsilon) - epsilon;
  
endfunction
