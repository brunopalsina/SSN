// REAL-TMIE MOLECULAR DYNAMICS SIMULATION WITH 2D VISUALISATION

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <array>
#include <cmath>
#include <random>

// settings
const unsigned int SCR_WIDTH = 450;
const unsigned int SCR_HEIGHT = 450;

// Constants for Lennard-Jones potential in SI units
const double PI = 3.14159265358979323846;
const double epsilon = 0.00286 * 1.60218e-19; // Depth of the potential well in Joules
const double sigma = 0.35e-9;                // Distance at which the potential is zero in meters
const double dt = 1e-15;                      // Time step for integration in seconds
const double cMass = 1.9944733e-26;
const double Kb = 1.380649e-23;

// Global variables
std::array<double, 2> r1, r2, r3, r4, r5, r6, r7, r8, r9; // r10;
double eq_dist = 3.6e-10;
double boxSize = 2.5*eq_dist;
double LH = eq_dist + 2*std::sqrt(0.75 * std::pow(eq_dist, 2));
double LW = 3*eq_dist;
double temperature = 297;
int Npart = 9;
double oldPositions[9][2] = {
	{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
	{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
};

// Shader source code
const char* vertexShaderSource = "#version 460 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";
const char* fragmentShaderSource = "#version 460 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
"}\n\0";

int main()
{
	//Initialize glfw library
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//Forward declarations
	void framebuffer_size_callback(GLFWwindow * window, int width, int height);
	void processInput(GLFWwindow * window);
	std::array<double, 39> dodecagonVertices(double centerX, double centerY);
	void integrate(std::array<double, 2>&r1, std::array<double, 2>&r2, std::array<double, 2>&r3, std::array<double, 2>&r4,
		std::array<double, 2>&r5, std::array<double, 2>&r6, std::array<double, 2>&r7, std::array<double, 2>&r8, std::array<double, 2>&r9, //std::array<double, 2>& r10,
		std::array<double, 2>&r1_old, std::array<double, 2>&r2_old, std::array<double, 2>&r3_old, std::array<double, 2>&r4_old,
		std::array<double, 2>&r5_old, std::array<double, 2>&r6_old, std::array<double, 2>&r7_old, std::array<double, 2>&r8_old, std::array<double, 2>&r9_old, // std::array<double, 2>& r10_old,
		std::array<double, 2>&a1, std::array<double, 2>&a2, std::array<double, 2>&a3, std::array<double, 2>&a4,
		std::array<double, 2>&a5, std::array<double, 2>&a6, std::array<double, 2>&a7, std::array<double, 2>&a8, std::array<double, 2>&a9); //, std::array<double, 2>& a10)
	std::array<double, 2> calculateForce(const std::array<double, 2>&r1);
	void vInitial();

	//Create a window object (pointer?) and prompt an error if it failed to do so
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Molecular Dynamics Real-time Simulation", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	//Prompt error if Glad failed to initialize
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//Define the location and size of the rendering window
	glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
	//Read modifications of the size of the 'window' (first argument) and call the defined function. in this case, the function sets the window size to the readings.
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	int success;
	char infoLog[512];
	//Shaders
	//Vertex shader
	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		return 0;
	}
	//Fragment shader
	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		return 0;
	}
	//Shader program. Final linked verion of multiple shaders combined
	unsigned int shaderProgram;
	shaderProgram = glCreateProgram(); //Returns the ID reference to the created object
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) 
	{
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::COMPILATION_FAILED\n" << infoLog << std::endl;
		return 0;
	}
	//Delete individual shaders as we have them now linked to the main one
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	//Initial parameters of the MD simulation
	//Initial positions of the particles in meters
	double y_from_center = std::sqrt(0.75 * std::pow(eq_dist, 2));
	r1 = { -1.25*eq_dist, y_from_center };
	r2 = { -0.25* eq_dist, y_from_center };
	r3 = { 0.75* eq_dist, y_from_center };
	r4 = { -0.75* eq_dist, 0.0 };
	r5 = { 0.25* eq_dist, 0.0 };
	r6 = { 1.25* eq_dist, 0.0 };
	r7 = { -1.25* eq_dist, -y_from_center };
	r8 = { -0.25* eq_dist, -y_from_center };
	r9 = { 0.75* eq_dist, -y_from_center };
	//r10 = { 0, 2*3.8e-10 };
	oldPositions[0][0] = r1[0];
	oldPositions[0][1] = r1[1];
	oldPositions[1][0] = r2[0];
	oldPositions[1][1] = r2[1];
	oldPositions[2][0] = r3[0];
	oldPositions[2][1] = r3[1];
	oldPositions[3][0] = r4[0];
	oldPositions[3][1] = r4[1];
	oldPositions[4][0] = r5[0];
	oldPositions[4][1] = r5[1];
	oldPositions[5][0] = r6[0];
	oldPositions[5][1] = r6[1];
	oldPositions[6][0] = r7[0];
	oldPositions[6][1] = r7[1];
	oldPositions[7][0] = r8[0];
	oldPositions[7][1] = r8[1]; 
	oldPositions[8][0] = r9[0];
	oldPositions[8][1] = r9[1];
	//oldPositions[9][0] = r10[0];
	//oldPositions[9][1] = r10[1];

	vInitial();

	std::array<double, 2> r1_old = { oldPositions[0][0], oldPositions[0][1] };
	std::array<double, 2> r2_old = { oldPositions[1][0], oldPositions[1][1] };
	std::array<double, 2> r3_old = { oldPositions[2][0], oldPositions[2][1] };
	std::array<double, 2> r4_old = { oldPositions[3][0], oldPositions[3][1] };
	std::array<double, 2> r5_old = { oldPositions[4][0], oldPositions[4][1] };
	std::array<double, 2> r6_old = { oldPositions[5][0], oldPositions[5][1] };
	std::array<double, 2> r7_old = { oldPositions[6][0], oldPositions[6][1] };
	std::array<double, 2> r8_old = { oldPositions[7][0], oldPositions[7][1] };
	std::array<double, 2> r9_old = { oldPositions[8][0], oldPositions[8][1] };
	//std::array<double, 2> r10_old = { oldPositions[9][0], oldPositions[9][1] };

	//Initial acceleration
	std::array<double, 2> force = calculateForce(r1);
	std::array<double, 2> a1 = { force[0] / cMass , (force[1] / cMass) };
	force = calculateForce(r2);
	std::array<double, 2> a2 = { force[0] / cMass , (force[1] / cMass) };
	force = calculateForce(r3);
	std::array<double, 2> a3 = { force[0] / cMass , (force[1] / cMass) };
	force = calculateForce(r4);
	std::array<double, 2> a4 = { force[0] / cMass , (force[1] / cMass) };
	force = calculateForce(r5);
	std::array<double, 2> a5 = { force[0] / cMass , (force[1] / cMass) };
	force = calculateForce(r6);
	std::array<double, 2> a6 = { force[0] / cMass , (force[1] / cMass) };
	force = calculateForce(r7);
	std::array<double, 2> a7 = { force[0] / cMass , (force[1] / cMass) };
	force = calculateForce(r8);
	std::array<double, 2> a8 = { force[0] / cMass , (force[1] / cMass) };
	force = calculateForce(r9);
	std::array<double, 2> a9 = { force[0] / cMass , (force[1] / cMass) };
	//force = calculateForce(r10);
	//std::array<double, 2> a10 = { force[0] / cMass , (force[1] / cMass) };

	//Geometry and vertex buffer assignation
	std::array<double, 39> verticesArray1 = dodecagonVertices(r1[0], r1[1]);
	std::array<double, 39> verticesArray2 = dodecagonVertices(r2[0], r2[1]);
	std::array<double, 39> verticesArray3 = dodecagonVertices(r3[0], r3[1]);
	std::array<double, 39> verticesArray4 = dodecagonVertices(r4[0], r4[1]);
	std::array<double, 39> verticesArray5 = dodecagonVertices(r5[0], r5[1]);
	std::array<double, 39> verticesArray6 = dodecagonVertices(r6[0], r6[1]);
	std::array<double, 39> verticesArray7 = dodecagonVertices(r7[0], r7[1]);
	std::array<double, 39> verticesArray8 = dodecagonVertices(r8[0], r8[1]);
	std::array<double, 39> verticesArray9 = dodecagonVertices(r9[0], r9[1]);
	//std::array<double, 39> verticesArray10 = dodecagonVertices(r10[0], r10[1]);

	float vertices[39*9];
	std::copy(verticesArray1.begin(), verticesArray1.end(), vertices);
	std::copy(verticesArray2.begin(), verticesArray2.end(), vertices + 39);
	std::copy(verticesArray3.begin(), verticesArray3.end(), vertices + 39 * 2);
	std::copy(verticesArray4.begin(), verticesArray4.end(), vertices + 39 * 3);
	std::copy(verticesArray5.begin(), verticesArray5.end(), vertices + 39 * 4);
	std::copy(verticesArray6.begin(), verticesArray6.end(), vertices + 39 * 5);
	std::copy(verticesArray7.begin(), verticesArray7.end(), vertices + 39 * 6);
	std::copy(verticesArray8.begin(), verticesArray8.end(), vertices + 39 * 7);
	std::copy(verticesArray9.begin(), verticesArray9.end(), vertices + 39 * 8);
	//std::copy(verticesArray10.begin(), verticesArray10.end(), vertices + 39 * 9);
	unsigned int indices[12*3*9]{0};
	int iteration{ 0 };
	for (int i = 0; i < 9; i++)
	{
		for (int a = i * 13; a < (i+1)*13 - 1; a++)
		{
			indices[iteration] = i * 13;
			indices[iteration + 1] = a + 1;
			if (a + 2 == ((i + 1) * 13))
			{
				indices[iteration + 2] = i * 13 + 1;
			}
			else
			{
				indices[iteration + 2] = a + 2;
			};
			//std::cout << "(" << indices[iteration] << "," << indices[iteration + 1] << "," << indices[iteration + 2] << ")" << std::endl;
			iteration += 3;
		};
	};
	//Generate and bind buffers
	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_DYNAMIC_DRAW);
	//Define how opengl should interpret the vertex data. Read docs for info on arguments.
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	//We can now unbind the array buffer from the VBO object
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Also possible to unbind the VAO so we dont accidentally modify it. Its is rare though.
	glBindVertexArray(0);

	//MAIN LOOP
	//Loop of lines to call each iteration (not yet aware of the loop's frequency) while the window is open
	int timefs = 0;
	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		
		//Rendering commands
		glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shaderProgram);
		glEnableVertexAttribArray(0);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 39 * 9, GL_UNSIGNED_INT, 0);
		//glBindVertexArray(0);

		if (timefs % 100000 == 0)
		{
			std::cout << "Time: " << timefs / 100000 << " picoseconds" << std::endl;
		};
		timefs += 1;

		integrate(r1, r2, r3, r4, r5, r6, r7, r8, r9, //r10,
			r1_old, r2_old, r3_old, r4_old, r5_old, r6_old, r7_old, r8_old, r9_old, //r10_old,
			a1, a2, a3, a4, a5, a6, a7, a8, a9); //a10);
		// Update accelerations based on new positions
		force = calculateForce(r1);
		a1 = { force[0] / cMass , (force[1] / cMass) };
		force = calculateForce(r2);
		a2 = { force[0] / cMass , (force[1] / cMass) };
		force = calculateForce(r3);
		a3 = { force[0] / cMass , (force[1] / cMass) };
		force = calculateForce(r4);
		a4 = { force[0] / cMass , (force[1] / cMass) };
		force = calculateForce(r5);
		a5 = { force[0] / cMass , (force[1] / cMass) };
		force = calculateForce(r6);
		a6 = { force[0] / cMass , (force[1] / cMass) };
		force = calculateForce(r7);
		a7 = { force[0] / cMass , (force[1] / cMass) };
		force = calculateForce(r8);
		a8 = { force[0] / cMass , (force[1] / cMass) };
		force = calculateForce(r9);
		a9 = { force[0] / cMass , (force[1] / cMass) };
		//force = calculateForce(r10);
		//a10 = { force[0] / cMass , (force[1] / cMass) };
		//std::cout << "(" << r1[0] << "," << r1[1] << ")" << std::endl;

		verticesArray1 = dodecagonVertices(r1[0], r1[1]);
		verticesArray2 = dodecagonVertices(r2[0], r2[1]);
		verticesArray3 = dodecagonVertices(r3[0], r3[1]);
		verticesArray4 = dodecagonVertices(r4[0], r4[1]);
		verticesArray5 = dodecagonVertices(r5[0], r5[1]);
		verticesArray6 = dodecagonVertices(r6[0], r6[1]);
		verticesArray7 = dodecagonVertices(r7[0], r7[1]);
		verticesArray8 = dodecagonVertices(r8[0], r8[1]);
		verticesArray9 = dodecagonVertices(r9[0], r9[1]);
		//verticesArray10 = dodecagonVertices(r10[0], r10[1]);

		std::copy(verticesArray1.begin(), verticesArray1.end(), vertices);
		std::copy(verticesArray2.begin(), verticesArray2.end(), vertices + 39);
		std::copy(verticesArray3.begin(), verticesArray3.end(), vertices + 39 * 2);
		std::copy(verticesArray4.begin(), verticesArray4.end(), vertices + 39 * 3);
		std::copy(verticesArray5.begin(), verticesArray5.end(), vertices + 39 * 4);
		std::copy(verticesArray6.begin(), verticesArray6.end(), vertices + 39 * 5);
		std::copy(verticesArray7.begin(), verticesArray7.end(), vertices + 39 * 6);
		std::copy(verticesArray8.begin(), verticesArray8.end(), vertices + 39 * 7);
		std::copy(verticesArray9.begin(), verticesArray9.end(), vertices + 39 * 8);
		//std::copy(verticesArray10.begin(), verticesArray10.end(), vertices + 39 * 9);
		
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_DYNAMIC_DRAW);
		//Define how opengl should interpret the vertex data. Read docs for info on arguments.
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		//We can now unbind the array buffer from the VBO object
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//Also possible to unbind the VAO so we dont accidentally modify it. Its is rare though.
		glBindVertexArray(0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glfwTerminate();

	return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

//Dodecagon vertices generator
std::array<double, 39> dodecagonVertices(double centerX, double centerY) 
{
	std::array<double, 39> vertices;
	double radius = (3.5e-10 / 2) / boxSize;

	// Dodecagon center coordinates
	vertices[0] = centerX / boxSize;
	vertices[1] = centerY / boxSize;
	vertices[2] = 0.0;

	// Coordinates of the 12 vertices
	for (int i = 0; i < 12; ++i) {
		double angle = 2 * PI * i / 12; // Calculate the angle for each vertex
		double x = centerX / boxSize + radius * std::cos(angle); // Calculate x-coordinate
		double y = centerY / boxSize + radius * std::sin(angle); // Calculate y-coordinate
		vertices[3 * (i + 1)] = x;
		vertices[3 * (i + 1) + 1] = y;
		vertices[3 * (i + 1) + 2] = 0.0;
	}

	return vertices;
}

// MOLECULAR DYNAMICS FUNCTIONS
// Lennard-Jones force
std::array<double, 2> calculateForce(const std::array<double, 2>& r) {
	std::array<double, 2> force = { 0.0, 0.0 };
	
	double positions[9][2] = {
		{r1[0], r1[1]}, {r2[0], r2[1]}, {r3[0], r3[1]},
		{r4[0], r4[1]}, {r5[0], r5[1]}, {r6[0], r6[1]},
		{r7[0], r7[1]}, {r8[0], r8[1]}, {r9[0], r9[1]} // {r10[0], r10[1]}
	};

	for (int i = 0; i < 9; ++i) {
		if (positions[i][0] == r[0] && positions[i][1] == r[1]) {
			continue;
		}
		double dx = positions[i][0] - r[0];
		double dy = positions[i][1] - r[1];
		dx -= LW * std::round(dx / LW);
		dy -= LH * std::round(dy / LH);
		double d = std::sqrt(dx * dx + dy * dy);
		if(d == 0) continue;
		double r2_inv = 1.0 / (d * d);
		double r6_inv = r2_inv * r2_inv * r2_inv;
		double r12_inv = r6_inv * r6_inv;
		double f_mag = -48.0 * epsilon * (std::pow(sigma, 12) * r12_inv - 0.5 * std::pow(sigma, 6) * r6_inv) * r2_inv;
		force[0] += f_mag * dx;
		force[1] += f_mag * dy;
		//std::cout << "(" << force[0] << "," << force[1] << ")" << std::endl;
	};
	//std::cout << "(" << force[0] << "," << force[1] << ")" << std::endl;
	return force;
}
void applyPBC(std::array<double, 2>& r) {
	r[0] -= LW * std::floor(r[0] / LW + 0.5);  // Adjust for centered origin
	r[1] -= LH * std::floor(r[1] / LH + 0.5);  // Adjust for centered origin
}
// Verlet integration
void integrate(std::array<double, 2>& r1, std::array<double, 2>& r2, std::array<double, 2>& r3, std::array<double, 2>& r4,
	std::array<double, 2>& r5, std::array<double, 2>& r6, std::array<double, 2>& r7, std::array<double, 2>& r8, std::array<double, 2>& r9, //std::array<double, 2>& r10,
	std::array<double, 2>& r1_old, std::array<double, 2>& r2_old, std::array<double, 2>& r3_old, std::array<double, 2>& r4_old,
	std::array<double, 2>& r5_old, std::array<double, 2>& r6_old, std::array<double, 2>& r7_old, std::array<double, 2>& r8_old, std::array<double, 2>& r9_old, // std::array<double, 2>& r10_old,
	std::array<double, 2>& a1, std::array<double, 2>& a2, std::array<double, 2>& a3, std::array<double, 2>& a4,
	std::array<double, 2>& a5, std::array<double, 2>& a6, std::array<double, 2>& a7, std::array<double, 2>& a8, std::array<double, 2>& a9) //, std::array<double, 2>& a10)
{
	std::array<double, 2> r1_new, r2_new, r3_new, r4_new, r5_new, r6_new, r7_new, r8_new, r9_new, r10_new;

	for (int i = 0; i < 2; ++i) {
		r1_new[i] = 2 * r1[i] - r1_old[i] + a1[i] * dt * dt;
		r2_new[i] = 2 * r2[i] - r2_old[i] + a2[i] * dt * dt;
		r3_new[i] = 2 * r3[i] - r3_old[i] + a3[i] * dt * dt;
		r4_new[i] = 2 * r4[i] - r4_old[i] + a4[i] * dt * dt;
		r5_new[i] = 2 * r5[i] - r5_old[i] + a5[i] * dt * dt;
		r6_new[i] = 2 * r6[i] - r6_old[i] + a6[i] * dt * dt;
		r7_new[i] = 2 * r7[i] - r7_old[i] + a7[i] * dt * dt;
		r8_new[i] = 2 * r8[i] - r8_old[i] + a8[i] * dt * dt;
		r9_new[i] = 2 * r9[i] - r9_old[i] + a9[i] * dt * dt;
		//r10_new[i] = 2 * r10[i] - r10_old[i] + a10[i] * dt * dt;
	}

	r1_old = r1;
	r2_old = r2;
	r3_old = r3;
	r4_old = r4;
	r5_old = r5;
	r6_old = r6;
	r7_old = r7;
	r8_old = r8;
	r9_old = r9;
	//r10_old = r10;
	r1 = r1_new;
	r2 = r2_new;
	r3 = r3_new;
	r4 = r4_new;
	r5 = r5_new;
	r6 = r6_new;
	r7 = r7_new;
	r8 = r8_new;
	r9 = r9_new;
	//r10 = r10_new;

	applyPBC(r1);
	applyPBC(r2);
	applyPBC(r3);
	applyPBC(r4);
	applyPBC(r5);
	applyPBC(r6);
	applyPBC(r7);
	applyPBC(r8);
	applyPBC(r9);
	//applyPBC(r10);
}
//Initial velocities function
void vInitial() {
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_real_distribution<double> distr(0.0, std::sqrt(0.5));
	double randValue, randValue2;

	double v0[9][2] = {
		{0.0,0.0},
		{0.0,0.0},
		{0.0,0.0},
		{0.0,0.0},
		{0.0,0.0},
		{0.0,0.0},
		{0.0,0.0},
		{0.0,0.0},
		{0.0,0.0}
	};
	double vCenterMass[2] = { 0.0, 0.0 };
	double kE[2] = { 0.0, 0.0 };
	for (int i = 0; i < 9; i++) {
		// Generate a random double between 0 and 1
		randValue = distr(eng);
		randValue2 = distr(eng);
		v0[i][0] = randValue;
		v0[i][1] = randValue2;
		vCenterMass[0] += v0[i][0];
		vCenterMass[1] += v0[i][1];
		kE[0] += cMass * std::pow(v0[i][0], 2);
		kE[1] += cMass * std::pow(v0[i][1], 2);
	};
	vCenterMass[0] = vCenterMass[0] / 9;
	vCenterMass[1] = vCenterMass[1] / 9;
	kE[0] = kE[0] / 9;
	kE[1] = kE[1] / 9;
	double scaleFactor = std::sqrt( 2 * Kb * temperature / std::sqrt(std::pow(kE[0], 2) + std::pow(kE[1], 2)));
	for (int i = 0; i < 9; i++) {
		v0[i][0] = ( v0[i][0] - vCenterMass[0]) * scaleFactor;
		v0[i][1] = ( v0[i][1] - vCenterMass[1]) * scaleFactor;
		oldPositions[i][0] += (-dt * v0[i][0]);
		oldPositions[i][1] += (-dt * v0[i][1]);
		//std::cout << "(" << oldPositions[i][0] << "," << oldPositions[i][1] << ")" << std::endl;
	};
};