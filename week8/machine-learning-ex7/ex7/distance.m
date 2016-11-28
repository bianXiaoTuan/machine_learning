function dis = distance(x, centroid)
	dis = sum((x - centroid) .^ 2);
end
