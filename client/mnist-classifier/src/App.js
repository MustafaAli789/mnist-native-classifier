
import Canvas from './Components/Canvas'
import PredictionBar from './Components/PredictionBar'
import { Button } from '@material-ui/core';
import Grid from '@material-ui/core/Grid';
import SearchIcon from '@material-ui/icons/Search';
import ClearIcon from '@material-ui/icons/Clear';
import './App.css'
import axios from 'axios'
import { useState } from 'react';


import createMixins from '@material-ui/core/styles/createMixins';

function App() {
  let centerStyle = {display: 'flex', alignItems: 'center', justifyContent: 'center'}

  const [preds, setPreds] = useState([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0] ])

  const clearCanvas = () => {
    let canvas = document.getElementById("canvas")
    let ctx = canvas.getContext("2d")
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  const classifyNum = () => {

    let canvas = document.getElementById("canvas")

    let img = document.getElementById("image");
    img.onload = function() {
      let canvasResized = document.createElement("canvas");
      canvasResized.width = 28;
      canvasResized.height = 28;
      let ctxResized = canvasResized.getContext("2d");
      ctxResized.drawImage(img, 0, 0, 28, 28);

      let imgd = ctxResized.getImageData(0, 0, 28, 28);
      let pix = imgd.data;
      let pixAlphaData = []
       
      // Loop over each pixel and invert the color.
      for (var i = 3, n = pix.length; i < n; i += 4) {
        pixAlphaData.push(pix[i])
      }
      console.log(pixAlphaData)

      axios.post('/api/classify', {"pixelMatrix": pixAlphaData})
        .then(response => {
          alert("The prediction is: " + response.data.pred[0])
          setPreds(response.data.preds)
        });

    }
    img.setAttribute("src", canvas.toDataURL())
  }

  return (
    <div style={{ maxWidth: '1000px' }}>
      <Grid container>
        <Grid style={centerStyle} item xs={12}>
          <h1>MNIST Classifier</h1>
        </Grid>
      </Grid>
      <Grid container>
        <Grid item xs={4}></Grid>
        <Grid style={{...centerStyle, flexDirection: 'column'}} item xs={4}>
          <Canvas />
          <div style={{flexDirection: 'row', width: '280px'}}>
            <Button
              style={{ width: '50%' }}
              variant="contained"
              color="primary"
              size="small"
              startIcon={<SearchIcon />}
              onClick={classifyNum}
            >
              Classify
            </Button>
            <Button
              style={{ width: '50%' }}
              variant="contained"
              color="secondary"
              size="small"
              startIcon={<ClearIcon />}
              onClick={clearCanvas}
            >
              Clear
            </Button>
          </div>
          <img id="image" src="" style={{display: "none"}}></img>
        </Grid>
        <Grid item xs={4}>
          <PredictionBar value={preds[0][0]} num={0}/>
          <PredictionBar value={preds[1][0]} num={1}/>
          <PredictionBar value={preds[2][0]} num={2}/>
          <PredictionBar value={preds[3][0]} num={3}/>
          <PredictionBar value={preds[4][0]} num={4}/>
          <PredictionBar value={preds[5][0]} num={5}/>
          <PredictionBar value={preds[6][0]} num={6}/>
          <PredictionBar value={preds[7][0]} num={7}/>
          <PredictionBar value={preds[8][0]} num={8}/>
          <PredictionBar value={preds[9][0]} num={9}/>
        </Grid>
      </Grid>
    </div>
  );
}

export default App;
