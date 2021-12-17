function im2 = filt2(kernel,im1,reflect_style)% im2 = filt2(kernel,im1,reflect_style)% Improved version of filter2, which includes reflection.% Default style is 'odd'.  Also can be 'even', or 'wrap'.% im2 = filt2(kern,image)	apply kern with odd refl (default).% im2 = filt2(kern,image,'even')	Use even reflection.% im2 = filt2(kern,image,128)	Fill with 128's.% Ruth Rosenholtzif nargin==2, reflect_style = 'odd'; end[ky,kx] = size(kernel);[iy,ix] = size(im1);imbig = addborder(im1,kx-1,ky-1,reflect_style);imbig = conv2(imbig,kernel,'same');im2 = imbig(ky:ky-1+iy, kx:kx-1+ix);