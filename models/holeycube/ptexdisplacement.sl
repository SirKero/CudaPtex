displacement ptexdisplacement(string mapname=""; float scale=0; varying float __faceindex=0)
{
    if (mapname != "") {
	point Pobj = transform("object", P);
	normal Nobj = normalize(transform("object", N));
	float disp = ptexture(mapname, 0, __faceindex, "filter", "bspline", "lerp", 1);
	Pobj += Nobj * scale * disp;
	P = transform("object", "current", Pobj);
	N = calculatenormal(P);
    }
}
