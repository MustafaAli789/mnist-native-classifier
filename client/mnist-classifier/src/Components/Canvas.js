import React, { useRef, useEffect, useState } from 'react';

function Canvas() {
    
    const [pos, _setPos] = useState({x:0,y:0})
    const posRef = useRef(pos)

    const setPos = data => {
        posRef.current = data
        _setPos(data)
    }

    function getMousePos(canvas, evt) {
        var rect = canvas.getBoundingClientRect();
        return {
          x: evt.clientX - rect.left,
          y: evt.clientY - rect.top
        };
    }

    const canvasRef = useRef(null)

    function draw(e) {
        // mouse left button must be pressed
        if(canvasRef === null) return
        let ctx = canvasRef.current.getContext("2d")

        if (e.buttons !== 1) return;
        ctx.beginPath(); // begin
        
        ctx.lineWidth = 25;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';
      
        ctx.moveTo(posRef.current.x, posRef.current.y); // from
        setPos(getMousePos(canvasRef.current, e));
        ctx.lineTo(posRef.current.x, posRef.current.y); // to
      
        ctx.stroke(); // draw it!
      }
    
    useEffect(() => {
        document.addEventListener('mousemove', draw);
        document.addEventListener('mousedown', (e) => setPos(getMousePos(canvasRef.current, e)));
        document.addEventListener('mouseenter', (e) => setPos(getMousePos(canvasRef.current, e)));
    }, []);

    return (
        <canvas ref={canvasRef} id="canvas" height="280" width="280"></canvas>
    )
}

export default Canvas;