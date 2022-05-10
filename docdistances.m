% part 1 find the tf-idf vector for each document

%load each document as character array
% make all words lower case for easier freq counting
RedRidingHood = lower(string(textread("RedRidingHood.txt", '%s')));
PrincessPea = lower(string(textread("PrincessPea.txt", '%s')));
Cinderella = lower(string(textread("Cinderella.txt", '%s')));
CAFA1 = lower(string(textread("CAFA1.txt", '%s')));
CAFA2 = lower(string(textread("CAFA2.txt", '%s')));
CAFA3 = lower(string(textread("CAFA3.txt", '%s')));

%%
%find the unique words in each document
RRH_uni = unique(RedRidingHood);
PP_uni = unique(PrincessPea);
C_uni = unique(Cinderella);
CAFA1_uni = unique(CAFA1);
CAFA2_uni = unique(CAFA2);
CAFA3_uni = unique(CAFA3);

%unique words in corpus 
text_uni = {RRH_uni, PP_uni, C_uni, CAFA1_uni, CAFA2_uni, CAFA3_uni}; % array containg words unique to specfic text
all_uni = unique([RRH_uni; PP_uni; C_uni; CAFA1_uni; CAFA2_uni; CAFA3_uni]); % array containg unique words in corpus

%%
%get the ivf vector for each document 

idf_RRH = idf(RedRidingHood, all_uni, text_uni);
idf_PP = idf(PrincessPea, all_uni, text_uni);
idf_C = idf(Cinderella, all_uni, text_uni);
idf_CAFA1 = idf(CAFA1, all_uni, text_uni);
idf_CAFA2 = idf(CAFA2, all_uni, text_uni);
idf_CAFA3 = idf(CAFA3, all_uni, text_uni);

% function for idf vector found at end of script 

%%
% Now a will create a 6*6 matrix containing the cosine distance between all vectors 

%First calc cos0 = a.b / |a||b| for every possible angle 

cos0 = zeros(6, 6); % zeros matrix to store all cos0
all_idf = {idf_RRH, idf_PP, idf_C, idf_CAFA1, idf_CAFA2, idf_CAFA3}; % cell array containing all idf vectors

for i = 1:6
    for j = 1:6
        vec1 = all_idf{i}; 
        vec2 = all_idf{j};
        cos0(i, j) = dot(vec1, vec2) / (norm(vec1) * norm(vec2));
    end
end

% Now calculate cosine distance = 1 - cos0
cosine_dist = 1 - cos0;

%%
im = imagesc(cosine_dist);
title('Cosine Distance');
colormap("bone")
colorbar






%%
function idfvect = idf(text, all_uni, text_uni)
no_uni = numel(all_uni); % number of unique words in corpus
term_freq = zeros(1, no_uni); % zero vector to store frequency of each term
occur_other = zeros(1, no_uni); % zero vector to store no of docs where term occurs

for i = 1:no_uni
    % count the occurance of each unique word 
    current_word = all_uni(i); % current word in loop 
    term_freq(i) = sum(count(text, current_word));
    
    for j = 1:6
        current_doc = text_uni{j}; % current string array contiang unique wrods of one doc in corpus 
        current_count =  sum(count(current_doc, current_word));
        if current_count > 0
            occur_other(i) = 1 + occur_other(i);
        end
    end
end

inverse_doc = log10(6 ./ occur_other); % calculate all inverse doc frequencies for idf vector
idfvect = term_freq .* inverse_doc; % calculate idf_vector
end