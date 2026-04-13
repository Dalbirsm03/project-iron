// Global Scene Setup
const container = document.getElementById('canvas-3d');
const video = document.getElementById('video-player');
const frameDisplay = document.getElementById('frame-display');

// Failsafe check
if (!container || !video) {
    console.error("CRITICAL ERROR: DOM elements not found. Check your HTML structure.");
}

const scene = new THREE.Scene();
// Add a helper grid so you can see the 3D space even if data fails
const gridHelper = new THREE.GridHelper(500, 50);
scene.add(gridHelper);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 2 / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth / 2, window.innerHeight);
container.appendChild(renderer.domElement);

// Center the camera based on the mock pipeline data boundaries 
// (X ranges 100-500, Y ranges 150-400)
camera.position.set(300, 275, 800); // Pull Z way back to widen the Field of View
camera.lookAt(300, 275, 0);         // Point exactly at the center of the data mass

const trajectoryLines = [];
const pointClouds = [];

function buildTrajectories(data) {
    data.tracks.forEach(track => {
        // 1. Build the Line Trajectory with Semantic Colors
        const points = [];
        const colors = [];

        track.path.forEach(pt => {
            // Push spatial coordinates
            points.push(new THREE.Vector3(pt.x, pt.y, pt.z));
            
            // Push the normalized PCA colors from the JSON
            // Failsafe: defaults to white if the python script failed to inject RGB
            colors.push(
                pt.r !== undefined ? pt.r : 1, 
                pt.g !== undefined ? pt.g : 1, 
                pt.b !== undefined ? pt.b : 1
            );
        });

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        // Inject the color buffer into the line geometry
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        // CRITICAL FIX: Remove 'color: 0x00ff00' and enable vertexColors
        const material = new THREE.LineBasicMaterial({ vertexColors: THREE.VertexColors, linewidth: 2 });
        const line = new THREE.Line(geometry, material);
        
        line.geometry.setDrawRange(0, 0); 
        scene.add(line);
        trajectoryLines.push({ line: line, data: track.path });

        // 2. Build the moving tracking dot with Semantic Colors
        const dotGeo = new THREE.BufferGeometry();
        dotGeo.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0], 3));
        // Initialize an empty color buffer for the dot
        dotGeo.setAttribute('color', new THREE.Float32BufferAttribute([1, 1, 1], 3)); 
        
        // CRITICAL FIX: Remove the hardcoded red and enable vertexColors
        const dotMat = new THREE.PointsMaterial({ size: 8, vertexColors: THREE.VertexColors });
        const dot = new THREE.Points(dotGeo, dotMat);
        scene.add(dot);
        pointClouds.push({ dot: dot, data: track.path });
    });
}

function sync3DWithVideo() {
    const currentFrame = Math.floor(video.currentTime * mockData.fps);
    frameDisplay.innerText = currentFrame;

    for (let i = 0; i < trajectoryLines.length; i++) {
        const lineObj = trajectoryLines[i];
        const dotObj = pointClouds[i];
        const path = lineObj.data;

        const targetIdx = Math.min(currentFrame, path.length - 1);

        if (targetIdx >= 0) {
            lineObj.line.geometry.setDrawRange(0, targetIdx);

            const currentPos = path[targetIdx];
            const positions = dotObj.dot.geometry.attributes.position.array;
            positions[0] = currentPos.x;
            positions[1] = currentPos.y;
            positions[2] = currentPos.z;
            dotObj.dot.geometry.attributes.position.needsUpdate = true;
        }
    }
}

video.addEventListener('timeupdate', sync3DWithVideo);
video.addEventListener('seeked', sync3DWithVideo);

let globalTrajectoryData = null;

// Fetch the real data converted from the Parquet file
fetch('trajectory_data.json?nocache=' + new Date().getTime())
    .then(response => {
        if (!response.ok) throw new Error("trajectory_data.json not found. Did you run data_converter.py?");
        return response.json();
    })
    .then(data => {
        globalTrajectoryData = data;
        buildTrajectories(data);
    })
    .catch(err => console.error("Data Loading Error:", err));

function sync3DWithVideo() {
    if (!globalTrajectoryData) return; // Prevent crashing before data loads

    const currentFrame = Math.floor(video.currentTime * globalTrajectoryData.fps);
    frameDisplay.innerText = currentFrame;

    for (let i = 0; i < trajectoryLines.length; i++) {
        const lineObj = trajectoryLines[i];
        const dotObj = pointClouds[i];
        const path = lineObj.data;

        const targetIdx = Math.min(currentFrame, path.length - 1);

        if (targetIdx >= 0) {
            lineObj.line.geometry.setDrawRange(0, targetIdx);

            const currentPos = path[targetIdx];
            const positions = dotObj.dot.geometry.attributes.position.array;
            positions[0] = currentPos.x;
            positions[1] = currentPos.y;
            positions[2] = currentPos.z;
            dotObj.dot.geometry.attributes.position.needsUpdate = true;
        }
    }
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();