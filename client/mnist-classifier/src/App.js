
import Canvas from './Components/Canvas'
import { Button } from '@material-ui/core';
import Grid from '@material-ui/core/Grid';
import SearchIcon from '@material-ui/icons/Search';
import './App.css'

function App() {
  let centerStyle = {display: 'flex', alignItems: 'center', justifyContent: 'center'}

  return (
    <div>
      <Grid container>
        <Grid style={centerStyle} item xs={12}>
          <h1>Title</h1>
        </Grid>
      </Grid>
      <Grid container>
        <Grid style={centerStyle} item xs={12}>
          <Canvas />
        </Grid>
        <Grid style={{...centerStyle, marginTop: '1rem'}} item xs={12}>
          <Button
            variant="contained"
            color="primary"
            size="small"
            startIcon={<SearchIcon />}
          >
            Classify
          </Button>
        </Grid>
      </Grid>
    </div>
  );
}

export default App;
