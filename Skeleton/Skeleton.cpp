//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Dézsenyi Balázs Zoltán
// Neptun : JJSFIL
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

//A kod nagy része megegyezik vagy épít a https://cg.iit.bme.hu/portal/oktatott-targyak/szamitogepes-grafika-es-kepfeldolgozas/geometriai-modellezes 
//linken talalhato kodokra
//ennek nagy része megtalálható az elõadásdiákban és ez egy tárgyhoz kapcsolódó oldal 
//(ezek alapján nem vagyok benne biztos hogy ezt meg kell jelölni)


#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

	void main() {
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform vec3 color;
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";




// 2D camera
struct Camera {
	float wCamx, wCamy;	// center in world coordinates
	float wWidth, wHeight;	// width and height in world coordinates
public:
	Camera() {
		wCamx = 0; 
		wCamy = 0;
		wWidth = 30;
		wHeight = 30;
	}

	mat4 View() { 
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCamx, -wCamy, 0, 1);
	}

	mat4 Projection() { 
		return mat4(2 / wWidth, 0, 0, 0,
			0, 2 / wHeight, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}


	mat4 ViewInverz() { 
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCamx, wCamy, 0, 1);
	}


	mat4 ProjectionInverz() { 
		return mat4(wWidth / 2, 0, 0, 0,
			0, wHeight / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	

	vec2 ScreenToWorld(float x, float y, int windowWidth, int windowHeight) {
		float ndcX = 2.0f * x / windowWidth - 1.0f;
		float ndcY = 1.0f - 2.0f * y / windowHeight;

		vec4 ndcCoords = vec4(ndcX, ndcY, 0.0f, 1.0f);
		vec4 worldCoords = ndcCoords * ViewInverz();

		worldCoords.x -= wCamx;
		worldCoords.y -= wCamy;

		vec2 worldPos = vec2(worldCoords.x, worldCoords.y);

		return worldPos;
	}

	
	void ZoomOut() {
		float t = 1.1f;
		wCamx *= t;
		wCamy *= t;
		wWidth *= t;
		wHeight *= t;
	}

	void ZoomIn() {
		float t = 1/1.1f;
		wCamx *= t;
		wCamy *= t;
		wWidth *= t;
		wHeight *= t;
	}

	void PanLeft() {
		wCamx -= 1.0f;
	}

	void PanRight() {
		wCamx += 1.0f;
	}



};


Camera camera;	

GPUProgram gpuProgram; 
const int nTesselatedVertices = 100;

class Curve {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;

protected:
	std::vector<vec4> wCP;		
	float distanceBetweenPoints(const vec4& p1, const vec4& p2) {
		float dx = p2.x - p1.x;
		float dy = p2.y - p1.y;
		float dz = p2.z - p1.z;
		float dw = p2.w - p1.w;

		return sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
	}
public:
	Curve() {
		// Curve
		glGenVertexArrays(1, &vaoCurve);
		glBindVertexArray(vaoCurve);

		glGenBuffers(1, &vboCurve); 
		glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
		glEnableVertexAttribArray(0); 
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);
		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		glEnableVertexAttribArray(0);  
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); 


	}

	virtual vec4 r(float t) { return wCP[0]; }
	virtual float tStart() { return 0; }
	virtual float tEnd() { return 1; }

	virtual void AddControlPoint(float cX, float cY) {
		vec4 calculated = vec4(cX, cY, 0, 1) * camera.ProjectionInverz() * camera.ViewInverz();
		wCP.push_back(calculated);
	}

	int PickControlPoint(float cX, float cY) {
		vec4 calculated = vec4(cX, cY, 0, 1) * camera.ProjectionInverz() * camera.ViewInverz();
		for (unsigned int p = 0; p < wCP.size(); p++) {
			if (dot(wCP[p] - calculated, wCP[p] - calculated) < 0.1) return p;
		}
		return -1;
	}

	void MoveControlPoint(int p, float cX, float cY) {
		vec4 calculated = vec4(cX, cY, 0, 1) * camera.ProjectionInverz() * camera.ViewInverz();
		wCP[p] = calculated;
	}

	void Draw() {
		mat4 VPTransform = camera.View() * camera.Projection();

		gpuProgram.setUniform(VPTransform, "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");



		if (wCP.size() > 0) {	
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCP.size() * 4 * sizeof(float), &wCP[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCP.size());
		}

		if (wCP.size() >= 2) {	
			std::vector<float> vertexData;
			for (int i = 0; i < nTesselatedVertices; i++) {	
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec4 wVertex = r(t);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
			}
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 0);
			glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);

		}
	}
};

class BezierCurve : public Curve {
	float B(int i, float t) {
		int n = wCP.size() - 1; 
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}
public:
	virtual vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 1);
		for (unsigned int n = 0; n < wCP.size(); n++) wPoint += wCP[n] * B(n, t);
		return wPoint;
	}
};


class LagrangeCurve : public Curve {
	std::vector<float> ts;  
	float L(int i, float t) {
		float Li = 1.0f;
		for (unsigned int j = 0; j < wCP.size(); j++)
			if ((int)j != i) Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}


public:
	void AddControlPoint(float cX, float cY) override {
		if (wCP.size() > 0) {
			float lastKnot = ts.back();
			float newKnot = lastKnot + distanceBetweenPoints(wCP.back(), vec4(cX, cY, 0, 1));
			ts.push_back(newKnot);
		}
		else {
			ts.push_back(0.0f);
		}
		Curve::AddControlPoint(cX, cY);
	}

	float tStart() { return ts[0]; }
	float tEnd() { return ts[wCP.size() - 1]; }

	virtual vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 0);
		for (unsigned int n = 0; n < wCP.size(); n++) wPoint += wCP[n] * L(n, t);
		return wPoint;
	}
};

float tension = 0;


class CatmullRomCurve : public Curve {
private:
	std::vector<float> ts;

	vec4 Hermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
		float t1_ = t - t0;
		float t2 = (t - t0) * (t - t0);
		float t3 = (t - t0) * (t - t0) * (t - t0);
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = (3.0f * (p1 - p0) / ((t1 - t0) * (t1 - t0))) - (2.0f * v0 + v1) / (t1 - t0);
		vec4 a3 = (2.0f * (p0 - p1) / ((t1 - t0) * (t1 - t0) * (t1 - t0))) + (v0 + v1) / ((t1 - t0) * (t1 - t0));

		return a0 + a1 * t1_ + a2 * t2 + a3 * t3;
	}
	


public:
	float tStart() override { return ts[0]; }
	float tEnd() override { return ts[wCP.size() - 1]; }

	void AddControlPoint(float cX, float cY) override {
		if (wCP.size() > 0) {
			float lastKnot = ts.back();
			float newKnot = lastKnot + distanceBetweenPoints(wCP.back(), vec4(cX, cY, 0, 1));
			ts.push_back(newKnot);
		}
		else {
			ts.push_back(0.0f);
		}
		Curve::AddControlPoint(cX, cY);
	}





	vec4 r(float t) override {
		for (unsigned int i = 0; i < wCP.size() - 1; i++) {
			if (ts[i] <= t && t <= ts[i + 1]) {
				unsigned int startI = i; unsigned int endI = i + 1;
				vec4 v0; vec4 v1;

				if (startI == 0 || startI == wCP.size() - 1) {
					v0 = vec4(0, 0, 0, 1);
				}
				else {
					v0 = 0.5 * (1 - tension) * (((wCP[startI + 1] - wCP[startI]) /
						(ts[startI + 1] - ts[startI])) + (wCP[startI] - wCP[startI - 1]) / (ts[startI] - ts[startI - 1]));
				}

				if (endI == 0 || endI == wCP.size() - 1) {
					v1 = vec4(0, 0, 0, 1);
				}
				else {
					v1 = 0.5 * (1 - tension) * (((wCP[endI + 1] - wCP[endI]) /
						(ts[endI + 1] - ts[endI])) + (wCP[endI] - wCP[endI - 1]) / (ts[endI] - ts[endI - 1]));
				}

				return vec4(Hermite(wCP[startI], v0, ts[startI], wCP[endI], v1, ts[endI], t));
			}
		}
		return vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

};




Curve* curve;



void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	curve = new Curve();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	curve->Draw();
	glutSwapBuffers();									
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'l') {
		curve = new LagrangeCurve();
	}
	else if (key == 'b') {
		curve = new BezierCurve();
	}
	else if (key == 'c') {
		curve = new CatmullRomCurve();
	}
	else if (key == 'Z') {
		camera.ZoomOut();
	}
	else if (key == 'z') {
		camera.ZoomIn();

	}
	else if (key == 'P') {
		camera.PanRight();
	}
	else if (key == 'p') {
		camera.PanLeft();
	}
	else if (key == 'T') {
		tension += 0.1f;
	}
	else if (key == 't') {
		tension -= 0.1f;
	}

	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

int pickedControlPoint = -1;


void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) { 
		vec2 worldCoords = camera.ScreenToWorld(pX, pY, windowWidth, windowHeight);

		curve->AddControlPoint(worldCoords.x, worldCoords.y);
		glutPostRedisplay();     
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {  
		pickedControlPoint = curve->PickControlPoint(cX, cY);
		glutPostRedisplay();     
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) { 
		pickedControlPoint = -1;
	}
}


void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (pickedControlPoint >= 0) curve->MoveControlPoint(pickedControlPoint, cX, cY);
}

void onIdle() {
	//long time = glutGet(GLUT_ELAPSED_TIME);
	glutPostRedisplay();
}