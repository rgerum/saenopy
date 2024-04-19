import styles from "./Logo.module.css"

export function Logo({width}) {
    return <img
        className={styles.logo}
        style={{"--width": width}}
        src={"https://saenopy.readthedocs.io/en/latest/_static/img/Logo_black.png"}
        alt={"Saenopy logo"} />
}
