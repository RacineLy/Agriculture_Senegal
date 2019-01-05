function g = gradsigmoid(z)
  
  var = sigmoid(z);
  g   = var.*(1 - var);
  
endfunction
