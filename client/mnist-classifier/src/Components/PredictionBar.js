import LinearProgress from '@material-ui/core/LinearProgress';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';

const PredictionBar = props => {
    return (
        <Box display="flex" alignItems="center" mr={3}>
            <Box>
              <Typography variant="body2" color="textSecondary">{props.num}</Typography>
            </Box>
            <Box width="100%" mr={1} ml={1}>
              <LinearProgress variant="determinate" value={Math.round(props.value*100)} />
            </Box>
            <Box minWidth={35}>
              <Typography variant="body2" color="textSecondary">{`${Math.round(
                props.value*100,
              )}%`}</Typography>
            </Box>
        </Box>
    )
}

export default PredictionBar