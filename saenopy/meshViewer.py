import numpy as np
import base64

def MeshViewer(R, L, F, U, f1, f2):
    source = """
    <!--<div id="info"><a href="http://threejs.org" target="_blank" rel="noopener">three.js</a> - dashed lines example</div>-->
    <div id="container"></div>
    
    <script src="https://threejs.org/build/three.js"></script>
    
    <script src="https://threejs.org/examples/js/WebGL.js"></script>
    <script src="https://threejs.org/examples/js/libs/stats.min.js"></script>
    <script src="https://threejs.org/examples/js/libs/dat.gui.min.js"></script>
    
    <script src="https://threejs.org/examples/js/controls/OrbitControls.js"></script>
    
    <style>
    
    .dg li {
        background: #f7f7f7 !important;
    }
    .dg {
       color: #111;
       text-shadow: none;
    }
    .dg.main .close-button {
        background: none;
    }
    .dg.main .close-button:hover {
       background: none;
    }
    .dg .cr.boolean {
        border-left: 1px solid #cfcfcf;
    }
    .dg .cr.number {
        border-left: 1px solid #cfcfcf;
    }
    .dg .c input[type=text] {
        background: #fffefe00;
        outline: none;
        color: #111 !important;
    }
    .dg .c input[type=text]:hover {
        background: #fffefe00;
        outline: none;
        color: #111 !important;
    }
    .dg .c .slider {
        background: #d6d6d6;
        cursor: ew-resize;
        border-radius: 5px;
    }
    .dg .c .slider:hover {
        background: #d6d6d6;
    }
    .dg .c .slider-fg {
        background: #747575;
        border-radius: 5px;
    }
    .dg .c .slider:hover .slider-fg {
       background: #42a5f5;
    }
    .dg li:not(.folder) {
        border: 1px solid #cfcfcf;
        border-radius: 2px;
    }
    
    </style>
    
    <script>
    
    function NewArray(type, base64) {
        var binary_string =  window.atob(base64);
        var len = binary_string.length;
        var bytes = new Uint8Array( len );
        for (var i = 0; i < len; i++)        {
            bytes[i] = binary_string.charCodeAt(i);
        }
        return new type(bytes.buffer);
    }
    
        //if ( WEBGL.isWebGLAvailable() === false ) {
        //    document.body.appendChild( WEBGL.getWebGLErrorMessage() );
        //}
    
        var renderer, scene, camera, stats, controls;
        var objects = [];
        var gui;
    
        factor_mesh = %f;
        factor_force = %f;
    
        var WIDTH = window.innerWidth, HEIGHT = window.innerHeight;
    
        init();
        animate();
    
        function init() {
    
            camera = new THREE.PerspectiveCamera( 60, WIDTH / HEIGHT, 1, 200 );
            camera.position.z = 150;
    
            scene = new THREE.Scene();
            scene.background = new THREE.Color( 0xFFFFFF);//0x111111 );
            scene.fog = new THREE.Fog( 0xFFFFFF, 50, 200);
    
            renderer = new THREE.WebGLRenderer( { antialias: true } );
            renderer.setPixelRatio( window.devicePixelRatio );
            renderer.setSize( WIDTH, HEIGHT );
    
            var container = document.getElementById( 'container' );
            container.appendChild( renderer.domElement );
    
            //stats = new Stats();
            //container.appendChild( stats.dom );
    
            //
            addMesh(%s, %s, %s, %s)
            window.addEventListener( 'resize', onWindowResize, false );
    
            controls = new THREE.OrbitControls( camera, renderer.domElement );
            //controls.minDistance = 10;
            //controls.maxDistance = 500;
            initGui();
    
        }
    
    
        function addMesh(points1, lines1, F1, U1) {
            points = points1;
            lines = lines1;
            F = F1;
            U = U1;
    
            for(var i=0; i < points.length; i++) {
                points[i] *= factor_mesh;
                U[i] *= factor_mesh;
            }
    
            //var h = size * 0.5;
    
            var geometry = new THREE.BufferGeometry();
            var position = [];
            //console.log(points.length, tets.length);
        
            for(var t=0; t < lines1.length/2; t++) {
                        var t1 = lines1[t*2+0];
                        var t2 = lines1[t*2+1];
                        for(var x=0; x < 3; x++)
                            position.push(points[t1*3+x]);
                        for(var x=0; x < 3; x++)
                            position.push(points[t2*3+x]);
                //console.log(t);
            }
            console.log("ready");
    
            geometry.addAttribute( 'position', new THREE.Float32BufferAttribute( position, 3 ) );
    
            //var geometryCube = cube( 50 );
    
            //var lineSegments = new THREE.LineSegments( geometry, new THREE.LineDashedMaterial( { color: 0xffaa00, dashSize: 3, gapSize: 1 } ) );
            mesh_lines = new THREE.LineSegments( geometry, new THREE.LineBasicMaterial( { color: 0xffaa00, linewidth: 0.5, transparent: true, opacity: 0.5} ) );
            mesh_lines.computeLineDistances();
    
            objects.push( mesh_lines );
            scene.add( mesh_lines );
    
            var geometry = new THREE.BufferGeometry();
            var position = [];
            var force_tips = [];
    
            for(var i=0; i < U.length/3; i++) {
                f_abs = Math.sqrt(F[i*3+0]**2 + F[i*3+1]**2 + F[i*3+2]**2);
                factor = factor_force*factor_mesh;//1/f_abs/3000 * f_abs * 100000;
                for(var x=0; x < 3; x++)
                    position.push((points[i*3+x]));
                for(var x=0; x < 3; x++) {
                    position.push(points[i * 3 + x] + F[i * 3 + x] * factor);
                    force_tips.push(points[i * 3 + x] + F[i * 3 + x] * factor);
                }
            }
    
            geometry.addAttribute( 'position', new THREE.Float32BufferAttribute( position, 3 ) );
    
            force_mat = new THREE.LineBasicMaterial( { color: 0xaa0000, linewidth: 3,} );
            force_lines = new THREE.LineSegments( geometry, force_mat );
            force_lines.computeLineDistances();
    
            objects.push( force_lines );
            scene.add( force_lines );
    
            var sprite = new THREE.TextureLoader().load( 'https://threejs.org/examples/textures/sprites/disc.png' );
    
            var geometry = new THREE.BufferGeometry();
            geometry.addAttribute( 'position', new THREE.Float32BufferAttribute( points, 3 ) );
            mesh_points = new THREE.Points( geometry, new THREE.PointsMaterial( { size: 8, sizeAttenuation: false, color: 0xffaa00, map: sprite, alphaTest: 0.5, transparent: true } ) );
            scene.add( mesh_points );
    
            var geometry = new THREE.BufferGeometry();
            geometry.addAttribute( 'position', new THREE.Float32BufferAttribute( force_tips, 3 ) );
            force_points = new THREE.Points( geometry, new THREE.PointsMaterial( { size: 10, sizeAttenuation: false, color: 0xaa0000, map: sprite, alphaTest: 0.5, transparent: true } ) );
            scene.add( force_points );
    
            // Displacements
    
            var geometry = new THREE.BufferGeometry();
            var position = [];
            var displacement_tips = [];
    
            for(var i=0; i < U.length/3; i++) {
                for(var x=0; x < 3; x++)
                    position.push((points[i*3+x]));
                for(var x=0; x < 3; x++) {
                    position.push(points[i * 3 + x] + U[i * 3 + x]);
                    displacement_tips.push(points[i * 3 + x] + U[i * 3 + x]);
                }
            }
    
            geometry.addAttribute( 'position', new THREE.Float32BufferAttribute( position, 3 ) );
            displacement_mat = new THREE.LineBasicMaterial( { color: 0x00aa00, linewidth: 2,} );
            displacement_lines = new THREE.LineSegments( geometry, displacement_mat );
            displacement_lines.computeLineDistances();
    
            objects.push( displacement_lines );
            scene.add( displacement_lines );
    
            var geometry = new THREE.BufferGeometry();
            geometry.addAttribute( 'position', new THREE.Float32BufferAttribute( displacement_tips, 3 ) );
            displacement_points = new THREE.Points( geometry, new THREE.PointsMaterial( { size: 10, sizeAttenuation: false, color: 0x00aa00, map: sprite, alphaTest: 0.5, transparent: true } ) );
            scene.add( displacement_points );
        }
    
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize( window.innerWidth, window.innerHeight );
        }
    
        function animate() {
            requestAnimationFrame( animate );
    
            render();
            renderer.render( scene, camera );
            //stats.update();
    
        }
    
        function render() {
    
            var time = Date.now() * 0.001;
    
            scene.traverse( function ( object ) {
    
                //if ( object.isLine ) {
    
                    //object.rotation.y = 0.25 * time;
                    //object.rotation.y = 0.25 * time;
    
                //}
    
            } );
    
            renderer.render( scene, camera );
    
        }
    
        function initGui() {
            gui = new dat.GUI();
            var param = {
                'mesh': true,
                'forces': true,
                'force scale': 1,
                'displacements': true,
                'view_range' : 200,
            };
            gui.add( param, 'mesh' ).onChange( function ( val ) {
                mesh_lines.visible = val;
                mesh_points.visible = val;
            } );
            gui.add( param, 'forces' ).onChange( function ( val ) {
                force_lines.visible = val;
                force_points.visible = val;
            } );
    
            gui.add( param, 'force scale', 1, 8, 0.1 ).onChange( function ( val ) {
                var position = [];
                var force_tips = [];
    
                for(var i=0; i < U.length/3; i++) {
                    f_abs = Math.sqrt(F[i * 3 + 0] ** 2 + F[i * 3 + 1] ** 2 + F[i * 3 + 2] ** 2);
                    factor = factor_force * factor_mesh * val;//1/f_abs/3000 * f_abs * 100000;
                    for (var x = 0; x < 3; x++)
                        position.push((points[i * 3 + x]));
                    for (var x = 0; x < 3; x++) {
                        position.push(points[i * 3 + x] + F[i * 3 + x] * factor);
                        force_tips.push(points[i * 3 + x] + F[i * 3 + x] * factor);
                    }
                }
                force_lines.geometry.addAttribute( 'position', new THREE.Float32BufferAttribute( position, 3 ) );
                force_points.geometry.addAttribute( 'position', new THREE.Float32BufferAttribute( force_tips, 3 ) );
            } );
    
            gui.add( param, 'displacements' ).onChange( function ( val ) {
                displacement_lines.visible = val;
                displacement_points.visible = val;
            } );
    
            gui.add( param, 'view_range', 10, 300, 1 ).onChange( function ( val ) {
                scene.fog.far = val;
            } );
        }
    
    </script>
    """
    source = source.replace("'", "\"")

    def wrap(array):
        if array.dtype == "float32":
            data_type = "Float32Array"
        elif array.dtype == "float64":
            data_type = "Float64Array"
        elif array.dtype == "int8":
            data_type = "Int8Array"
        elif array.dtype == "uint8":
            data_type = "Uint8Array"
        elif array.dtype == "int16":
            data_type = "Int16Array"
        elif array.dtype == "uint16":
            data_type = "Uint16Array"
        elif array.dtype == "int32":
            data_type = "Int32Array"
        elif array.dtype == "uint32":
            data_type = "Uint32Array"
        elif array.dtype == "int64":
            data_type = "BigInt64Array"
        elif array.dtype == "uint64":
            data_type = "BigUint64Array"
        else:
            raise TypeError(array.dtype)
        return "NewArray("+data_type+", \""+repr(base64.b64encode(array))[2:-1]+"\")"

    here = source % (f1, f2, wrap(R-np.mean(R, axis=0)), wrap(L), wrap(F), wrap(U))
    from IPython.core.display import HTML, display
    code = "<h1></h1><iframe srcdoc='{0}' scrolling=no style='border:none; width: 100%; height: 600px'></iframe>".format(here)
    display(HTML(code))
