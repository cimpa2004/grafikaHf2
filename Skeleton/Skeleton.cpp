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
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}


	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}


	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}

	vec2 ScreenToWorld(float x, float y, int windowWidth, int windowHeight) {
		float ndcX = 2.0f * x / windowWidth - 1.0f;
		float ndcY = 1.0f - 2.0f * y / windowHeight;

		vec4 ndcCoords = vec4(ndcX, ndcY, 0.0f, 1.0f);
		vec4 worldCoords = ndcCoords * Vinv();

		worldCoords.x -= wCx;
		worldCoords.y -= wCy;

		vec2 worldPos = vec2(worldCoords.x, worldCoords.y);

		return worldPos;
	}

	void Zoom(float zoom) {
		float converted = 1;
		if (zoom == 1){
			return;
		}
		else if (zoom < 1) {
			converted = 0.9f;
		}
		else if ( zoom > 1) {
			converted = 1.1f;
		}
		wCx *= converted;
		wCy *= converted;
		wWx *= converted;
		wWy *= converted;
		
	}

	void PanLeft() {
		wCx -= 1.0f;
	}

	void PanRight() {
		wCx += 1.0f;
	}



};


Camera camera;	// 2D camera

GPUProgram gpuProgram; // vertex and fragment shaders
const int nTesselatedVertices = 100;
float currentZoom = 1;

class Curve {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
protected:
	std::vector<vec4> wCtrlPoints;		// coordinates of control points
public:
	Curve() {
		// Curve
		glGenVertexArrays(1, &vaoCurve);
		glBindVertexArray(vaoCurve);

		glGenBuffers(1, &vboCurve); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

		// Control Points
		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset


	}

	virtual vec4 r(float t) { return wCtrlPoints[0]; }
	virtual float tStart() { return 0; }
	virtual float tEnd() { return 1; }

	virtual void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints.push_back(wVertex);
	}
	



	// Returns the selected control point or -1
	int PickControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		for (unsigned int p = 0; p < wCtrlPoints.size(); p++) {
			if (dot(wCtrlPoints[p] - wVertex, wCtrlPoints[p] - wVertex) < 0.1) return p;
		}
		return -1;
	}

	void MoveControlPoint(int p, float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints[p] = wVertex;
	}

	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		gpuProgram.setUniform(VPTransform, "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

		

		if (wCtrlPoints.size() > 0) {	// draw control points
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}

		if (wCtrlPoints.size() >= 2) {	// draw curve
			std::vector<float> vertexData;
			for (int i = 0; i < nTesselatedVertices; i++) {	// Tessellate
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec4 wVertex = r(t);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
			}
			// copy data to the GPU
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 0);
			glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);

		}
	}
};

// Bezier using Bernstein polynomials
class BezierCurve : public Curve {
	float B(int i, float t) {
		int n = wCtrlPoints.size() - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}
public:
	virtual vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 0);
		for (unsigned int n = 0; n < wCtrlPoints.size(); n++) wPoint += wCtrlPoints[n] * B(n, t);
		return wPoint;
	}
};

// Lagrange curve
class LagrangeCurve : public Curve {
	std::vector<float> ts;  // knots
	float L(int i, float t) {
		float Li = 1.0f;
		for (unsigned int j = 0; j < wCtrlPoints.size(); j++)
			if (j != i) Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}
public:
	void AddControlPoint(float cX, float cY) {
		ts.push_back((float)wCtrlPoints.size());
		Curve::AddControlPoint(cX, cY);
	}
	float tStart() { return ts[0]; }
	float tEnd() { return ts[wCtrlPoints.size() - 1]; }

	virtual vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 0);
		for (unsigned int n = 0; n < wCtrlPoints.size(); n++) wPoint += wCtrlPoints[n] * L(n, t);
		return wPoint;
	}
};

float tension = 0;


class CatmullRom : public Curve {
private:
	std::vector<float> ts;
	std::vector<vec3> cps;

	vec3 Hermite(vec3 p0, vec3 v0, float t0, vec3 p1, vec3 v1, float t1, float t) {
		float t1_ = t - t0;
		float t2 = (t - t0) * (t - t0);
		float t3 = (t - t0) * (t - t0) * (t - t0);
		vec3 a0 = p0;
		vec3 a1 = v0;
		vec3 a2 = (3.0f * (p1 - p0) / ((t1 - t0) * (t1 - t0))) - (2.0f * v0 + v1) / (t1 - t0);
		vec3 a3 = (2.0f * (p0 - p1) / ((t1 - t0) * (t1 - t0) * (t1 - t0))) + (v0 + v1) / ((t1 - t0) * (t1 - t0));

		return a0 + a1 * t1_ + a2 * t2 + a3 * t3;
	}



public:
	float tStart() { return ts[0]; }
	float tEnd() { return ts[wCtrlPoints.size() -1 ]; }

	void AddControlPoint(float cX, float cY) {
		ts.push_back((float)wCtrlPoints.size());
		cps.push_back(vec3(cX, cY, 1));
		Curve::AddControlPoint(cX, cY);
	}

	virtual vec4 r(float t) override {
		for (int i = 0; i < cps.size() - 1; i++) {
			if (ts[i] <= t &&  t <= ts[i + 1]) {
				int startI = i; int endI = i + 1;
				vec3 v0; vec3 v1;

				if (startI == 0) {
					v0 = vec3(0, 0, 0);
					printf("AAAAAAAAAAAAAAAAAAAAAAAA   %d \n", startI);
				}
				else {
					v0 = 0.5 * (1 - tension) * (((cps[startI + 1] - cps[startI]) /
						(ts[startI + 1] - ts[startI])) + (cps[startI] - cps[startI - 1]) / (ts[startI] - ts[startI - 1]));
				}
					
				if (endI == cps.size() -1 ) {
					v1 = vec3(0, 0, 0);
					printf("BBBBBBBBBBBBBBBBBBBBBBBBBB  %d \n", endI);
					
				}
				else {
					v1 = 0.5 * (1 - tension) * (((cps[endI + 1] - cps[endI]) /
						(ts[endI + 1] - ts[endI])) + (cps[endI] - cps[endI - 1]) / (ts[endI] - ts[endI - 1]));
				}

				vec3 temp = Hermite(cps[startI], v0, ts[startI], cps[endI], v1, ts[endI], t);
				return vec4(temp.x, temp.y, temp.z, 1);
			}
		}
		return vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

};




Curve* curve;



// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	curve = new Curve();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	curve->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'l'){
		curve = new LagrangeCurve();
	}
	else if (key == 'b') {
		curve = new BezierCurve();
	}
	else if (key == 'c') {
		curve = new CatmullRom();
	}
	else if (key == 'Z') {
		currentZoom = 2;
		camera.Zoom(currentZoom);
	}
	else if (key == 'z') {
		currentZoom = 0;
		camera.Zoom(currentZoom);

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

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

int pickedControlPoint = -1;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		vec2 worldCoords = camera.ScreenToWorld(pX, pY, windowWidth, windowHeight);

		curve->AddControlPoint(worldCoords.x, worldCoords.y);
		glutPostRedisplay();     // redraw
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		pickedControlPoint = curve->PickControlPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		pickedControlPoint = -1;
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;	
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (pickedControlPoint >= 0) curve->MoveControlPoint(pickedControlPoint, cX, cY);
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program		
	glutPostRedisplay();
}