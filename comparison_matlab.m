original_path= 'dataset/BSD500/images/test';

generated_path = 'NN';


files= dir(original_path);
files=files(3:end);
sum=0;
psnr_value= 0;

for i=1:length(files)
    file_name=files(i).name;
    
    image=imread(strcat(original_path, '/', file_name));
    
    image_=imread(strcat(generated_path, '/', file_name));
    
    fsim= FeatureSIM(image, image_);
    
    [peaksnr, snr]=psnr(image, image_);
    sum= sum+fsim;
    psnr_value= psnr_value+ peaksnr;
end

fprintf('Final FSIM is %i', sum/100)
fprintf('Final PSNR is %i', psnr_value/100)