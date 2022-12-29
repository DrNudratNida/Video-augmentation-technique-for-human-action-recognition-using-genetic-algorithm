function confPlot(Conf1,order)
z=sum(Conf1');
prob = Conf1./(sum(Conf1')'*ones(1, size(Conf1,1)));
newProb = round(prob*100000000)/1000000;
for i=1:size(Conf1,2)
			for j=1:size(Conf1,2)
				if Conf1(i,j)==0
					strMat{i,j}='0';
				else
					strMat{i,j}=[ num2str(Conf1(i,j)), '/',num2str(z(i))];
				end
			end
end
ab=newProb;
for i=1:size(Conf1,2)
			for j=1:size(Conf1,2)
				if Conf1(i,j)==0
					strMat1{i,j}='0';
				else
					strMat1{i,j}=num2str(ab(i,j),'%0.1f');
				end
			end
end


accuracy=sum(diag(ab))/100;
mat=100-ab;
%in=input('\nPlease select option:\n 1 for Percentage based Confusion Matrix\n 2 for no./sum based Confusion Matrix\n');

in=1;
if in==1
    
        textStrings=strMat1(:); %# Create strings from the matrix values
        titleOfMatrix=' Confusion Matrix in %';
else
        textStrings = strMat(:);%
        titleOfMatrix='Confusion Matrix in correct/total elements of class';
end
figure,
imagesc(mat);            %# Create a colored plot of the matrix values
colormap('gray');
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:length(order));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(100-mat(:) > midValue,1,3); 
set(hStrings,{'Color'},num2cell(textColors,2));
set(gca,'XTick',1:length(order),...                         %# Change the axes tick marks
        'XTickLabel',order,...  %#   and tick labels
        'YTick',1:length(order),...
        'YTickLabel',order,...
        'TickLength',[0 0],'FontWeight', 'bold');
    %rotate text 
%get current tick labels
a=get(gca,'XTickLabel');
%erase current tick labels from figure
set(gca,'XTickLabel',[]);
%get tick label positions
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=30;
%make new tick labels
text(b,repmat(max(c)+1,length(b),1)-0.4,a,'HorizontalAlignment','right','rotation',rot,'FontSize', 10, 'FontWeight', 'bold');
% grid lines
    xlim = get(gca,'XLim');
    ylim = get(gca,'YLim');
    for i = 1:diff(xlim)-1
        line('Parent',gca,'XData',[i i]+.5, 'YData', ylim,'LineStyle','-');
    end
    for i = 1:diff(ylim)-1
        line('Parent',gca,'XData',xlim, 'YData', [i i]+.5,'LineStyle','-');
    end
    title(titleOfMatrix);